
from __future__ import absolute_import

import math
import random
import sys
import scipy
import imageio
import scipy.misc
import scipy.spatial
import scipy.ndimage
import numpy as np
import torch

import models

from typing import *


def plot_image_scatter(ax, data, images, downscaleRatio: Optional[int] = None):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    if downscaleRatio:
        ratioInv = 1 / downscaleRatio
        images = [scipy.ndimage.interpolation.zoom(i, [ratioInv, ratioInv, 1], order=1) for i in images]

    ax.scatter(data[:, 0], data[:, 1])

    for i in range(len(images)):
        x, y = tuple(data[i])
        ab = AnnotationBbox(OffsetImage(images[i]), (x, y), frameon=False)
        ax.add_artist(ab)


def generate_points(dataSize: int):
    modeNumber = 3
    modeRadius = 2.0
    std = 1.0 / 2
    means = []
    for i in range(modeNumber):
        phase = i / modeNumber * 2 * math.pi
        means.append([math.cos(phase) * modeRadius, math.sin(phase) * modeRadius])

    choices = [np.random.multivariate_normal(m, np.identity(2) * std, dataSize) for m in means]
    points = np.asarray([choices[random.randrange(modeNumber)][i] for i in range(dataSize)])

    return points


def main():
    use_gpu = True

    ref_path  = './imgs/ex_ref.png'
    pred_path = './imgs/ex_p1.png'

    ref_img = imageio.imread(ref_path).transpose(2, 0, 1) / 255.
    pred_img = imageio.imread(pred_path).transpose(2, 0, 1) / 255.

    gpu = torch.device('cuda')

    # Torchify
    # ref = torch.tensor(ref_img[np.newaxis, ...]).to(gpu)
    # pred = torch.tensor(pred_img[np.newaxis, ...], requires_grad=True).to(gpu)

    dataSize = 128
    batchSize = 8

    # Generate a random distance matrix.
    # # Make a matrix with positive values.
    # distancesCpu = np.clip(np.random.normal(0.5, 1.0 / 3, (dataSize, dataSize)), 0, 1)
    # # Make it symmetrical.
    # distancesCpu = np.matmul(distancesCpu, distancesCpu.T)

    # Generate random points and compute distances, guaranteeing that the triangle rule isn't broken.
    randomPoints = generate_points(dataSize)
    distancesCpu = scipy.spatial.distance_matrix(randomPoints, randomPoints, p=2)

    imagesInitCpu = np.clip(np.random.normal(0.5, 0.5 / 3, (dataSize, 3, 64, 64)), 0, 1)
    images = torch.tensor(imagesInitCpu, requires_grad=True, dtype=torch.float32, device=gpu)

    scale = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=gpu)

    lossModel = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=use_gpu).to(gpu)
    optimizer = torch.optim.Adam([images, scale], lr=1e-3, betas=(0.9, 0.999))

    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=batchSize // 2)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    for i in range(10000):

        # noinspection PyTypeChecker
        randomIndices = np.random.randint(0, dataSize, batchSize).tolist()  # type: List[int]
        # randomIndices = list(range(dataSize))  # type: List[int]
        distanceBatch = torch.tensor(distancesCpu[randomIndices, :][:, randomIndices], dtype=torch.float32, device=gpu)
        imageBatch = images[randomIndices].contiguous()

        distPred = lossModel.forward(imageBatch.repeat(repeats=(batchSize, 1, 1, 1)).contiguous(),
                                     imageBatch.repeat_interleave(repeats=batchSize, dim=0).contiguous(), normalize=True)
        distPredMat = distPred.reshape((batchSize, batchSize))

        loss = torch.sum((distanceBatch - distPredMat * scale) ** 2)  # MSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        images.data = torch.clamp(images.data, 0, 1)

        if i % 100 == 0:
            msg = 'iter {}, loss {:.3f}, scale: {:.3f}'.format(i, loss.item(), scale.item())
            print(msg)
            imageBatchCpu = imageBatch.cpu().data.numpy().transpose(0, 2, 3, 1)
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(imageBatchCpu[i])
            fig.suptitle(msg)

            imagesAllCpu = images.cpu().data.numpy().transpose(0, 2, 3, 1)
            plot_image_scatter(ax2, randomPoints, imagesAllCpu, downscaleRatio=2)

            plt.pause(5e-2)
            # plt.imsave('imgs_saved/%04d.jpg'%i,pred_img)


if __name__ == '__main__':
    main()
