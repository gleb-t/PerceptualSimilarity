"""
Started out as a copy-paste from the E-LPIPS Tensorflow repo.
"""
import itertools
import random

import imageio
import numpy as np
import torch
import torch.nn
import torch.random


class Config:
    def __init__(self):
        self.enable_offset = True
        self.offset_max = 7

        self.enable_flip = True
        self.enable_swap = True
        self.enable_color_permutation = True

        self.enable_color_multiplication = True
        self.color_multiplication_mode = 'color'  # 'brightness'

        self.enable_scale = True
        self.set_scale_levels(8)

        # Enables cropping instead of padding. Faster but may randomly skip edges of the input.
        self.fast_and_approximate = False

        self.batch_size = 1
        self.average_over = 1  # How many runs to average over.

        self.dtype = torch.float32

    def set_scale_levels(self, num_scales):
        # Crop_size / num_scales should be at least 64.
        self.num_scales = num_scales
        self.scale_probabilities = [1.0 / float(i) ** 2 for i in range(1, self.num_scales + 1)]

    def set_scale_levels_by_image_size(self, image_h, image_w):
        '''Sets the number of scale levels based on the image size.'''
        image_size = min(image_h, image_w)
        self.set_scale_levels(max(1, image_size // 64))


def sample_ensemble(config):
    """
    Some of these transformations are defined by sample in the batch,
    but those that can't be applied to tensor slices (resizing, transposition) are shared by the whole batch.
    Furthermore, some of the sampling is stratified (latin hypercube), thus the code is more complex
    that it would be for the basic independent sampling.
    """
    n = config.batch_size

    # Offset randomization.
    offset_xy = torch.randint(0, config.offset_max, (n, 2))

    # Sample scale level.
    cumulative_sum = np.cumsum(config.scale_probabilities)
    u = cumulative_sum[-1] * random.uniform(0, 1)
    scale_level = next((i for i, p in enumerate(cumulative_sum) if u < p), len(cumulative_sum))
    scale_level = max(min(scale_level, config.num_scales), 1)

    # Scale randomization.
    scale_offset_xy = torch.randint(0, scale_level, (2,))

    # Sample flips.
    flips = torch.arange(0, (n + 3) // 4 * 4, dtype=torch.int32)
    flips = flips % 4
    flips = flips[torch.randperm(flips.shape[0])]  # Shuffle.
    flips = flips[:n]

    # Sample transposing.
    # swap_xy = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)  # Disabled for now.
    swap_xy = 0

    # Color multiplication.
    def sample_colors() -> torch.Tensor:
        color = torch.rand((n,), dtype=config.dtype)
        color += torch.arange(0, n, dtype=config.dtype)
        color /= n

        return color[torch.randperm(n)]  # Shuffle.
    colors_r = sample_colors().view((-1, 1, 1, 1))
    colors_g = sample_colors().view((-1, 1, 1, 1))
    colors_b = sample_colors().view((-1, 1, 1, 1))

    if config.color_multiplication_mode == 'color':
        color_factors = torch.cat([colors_r, colors_g, colors_b], dim=1)
    elif config.color_multiplication_mode == 'brightness':
        color_factors = torch.cat([colors_r, colors_r, colors_r], dim=1)
    else:
        raise Exception('Unknown color multiplication mode.')

    color_factors = 0.2 + 0.8 * color_factors

    # Sample permutations.
    permutations = np.asarray(list(itertools.permutations(range(3))), dtype=np.int32)
    repeat_count = (n + len(permutations) - 1) // len(permutations)
    permutations = torch.tensor(permutations).repeat((repeat_count, 1))
    permutations = permutations[torch.randperm(permutations.shape[0])][:n, :].flatten()
    # I'm not sure why we needed that second dim there, but I'm staying close to the TF code for now.

    base_indices = 3 * torch.arange(0, n).unsqueeze(1).repeat((1, 3)).flatten()  # [0, 0, 0, 3, 3, 3, 6, 6, 6, ...]
    permutations += base_indices

    return offset_xy, flips, swap_xy, color_factors, permutations, scale_offset_xy, scale_level


def apply_ensemble(config, sampled_ensemble_params, X):
    offset_xy, flips, swap_xy, color_factors, permutations, scale_offset_xy, scale_level = sampled_ensemble_params

    N, C, H, W = X.shape

    # Resize image.
    if config.enable_scale and scale_level != 1:
        if config.fast_and_approximate:
            raise NotImplementedError()
        else:
            # Pad to a multiple of scale_level.
            pad_left = scale_offset_xy[1]
            full_width = (scale_level - 1 + W + scale_level - 1) // scale_level * scale_level
            pad_right = full_width - W - pad_left

            pad_bottom = scale_offset_xy[0]
            full_height = (scale_level - 1 + H + scale_level - 1) // scale_level * scale_level
            pad_top = full_height - H - pad_bottom

            X = torch.nn.functional.pad(X, [pad_left, pad_right, pad_bottom, pad_top], 'reflect')
            N, C, H, W = X.shape

        scaled = X.view((N, C, H // scale_level, scale_level, W // scale_level, scale_level))
        scaled = torch.mean(scaled, dim=[3, 5])

        X = scaled
        N, C, H, W = X.shape

    # Pad image.
    if config.enable_offset:
        paddedList = []

        for i in range(config.batch_size):
            if config.fast_and_approximate:
                raise NotImplementedError()
            else:
                # Pad.
                pad_bottom = config.offset_max - offset_xy[i, 0]
                pad_left = config.offset_max - offset_xy[i, 1]
                pad_top = offset_xy[i, 0]
                pad_right = offset_xy[i, 1]

                padded = torch.nn.functional.pad(X[i, ...].unsqueeze(0),  # torch.pad requires a 4D tensor.
                                                 [pad_left, pad_right, pad_bottom, pad_top], 'reflect')
                paddedList.append(padded)

        X = torch.cat(paddedList, dim=0)
        N, C, H, W = X.shape

    # Apply flips.
    if config.enable_flip:
        flippedList = []
        for i in range(config.batch_size):
            if flips[i] == 0:
                flippedList.append(X[i].flip(2))
            elif flips[i] == 1:
                flippedList.append(X[i].flip(1))
            elif flips[i] == 2:
                flippedList.append(X[i].flip(1, 2))
            else:
                flippedList.append(X[i, :, :, :])

        X = torch.stack(flippedList, dim=0)

    # Apply transpose.
    if config.enable_swap:
        if swap_xy:
            X = X.permute(0, 1, 3, 2)
            N, C, H, W = X.shape

    # Apply color permutations.
    if config.enable_color_permutation:
        # X = tf.transpose(X, [0, 3, 1, 2])  # NHWC -> NCHW
        # X = tf.reshape(X, [N * C, H, W])  # (NC)HW
        # X = tf.gather(X, perms)  # Permute rows (colors)
        # X = tf.reshape(X, [N, C, H, W])  # NCHW
        # X = tf.transpose(X, [0, 2, 3, 1])  # NCHW -> NHWC


        # X = X.gather(dim=1, index=permutations.long())  # todo Need to check if this works correctly.

        # todo I should re-implement this in a readable way, without gather. Would also save some memory.
        Y = X.view((N * C, H * W))
        Y = Y.gather(dim=0, index=permutations.long().unsqueeze(1).repeat((1, Y.shape[1])))
        Y = Y.view((N, C, H, W))
        # assert np.testing.assert_equal(X.numpy(), Y.numpy())

        X = Y

    if config.enable_color_multiplication:
        X = X * color_factors

    return X


if __name__ == '__main__':
    config = Config()
    config.batch_size = 10

    imagePath = r'E:\data\cat-vs-dog\cat\cat.1.jpg'
    imageData = imageio.imread(imagePath)[np.newaxis, :, :, :3].transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    config.set_scale_levels_by_image_size(imageData.shape[2], imageData.shape[3])

    imageBatch = torch.tensor(imageData).repeat((config.batch_size, 1, 1, 1))

    ensembleParams = sample_ensemble(config)
    offset_xy, flips, swap_xy, color_factors, permutations, scale_offset_xy, scale_level = ensembleParams

    imageBatchTransformed = apply_ensemble(config, ensembleParams, imageBatch)

    print(offset_xy)
    print(imageBatchTransformed.shape)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=config.batch_size)
    for i, ax in enumerate(axes):
        ax.imshow(imageBatchTransformed[i].numpy().transpose(1, 2, 0))

    plt.show()

    pass