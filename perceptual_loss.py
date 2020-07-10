import itertools
import math
import os
import random
import glob
import scipy
import imageio
import scipy.misc
import scipy.spatial
import scipy.ndimage
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import skimage.transform

import models

from typing import *


class CatDataset(Dataset):

    def __init__(self, imageSubdirPath: str, transform: Callable):
        self.rootPath = imageSubdirPath
        self.pathList = glob.glob(os.path.join(self.rootPath, 'cat*.jpg'))

        self.transform = transform

    def __getitem__(self, index):
        path = self.pathList[index]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(path)  # Pass image name as metadata. (Useful for export.)

    def __len__(self):
        return len(self.pathList)


class InfiniteSampler(Sampler):
    """
    From https://discuss.pytorch.org/t/implementing-an-infinite-loop-dataset-dataloader-combo/35567/3
    """
    def __init__(self, data_source: Sized):
        super().__init__(data_source)
        self.dataset_size = len(data_source)

    def __iter__(self):
        # This wrapper is only needed if we want to use a sample size other than one.
        # Otherwise, iteration over the random permutation tensors is already yielding single indices.
        yield from itertools.islice(self._infinite(), 0, None, 1)  # Infinite iterator

    def _infinite(self):
        g = torch.Generator()
        while True:
            yield from torch.randperm(self.dataset_size, generator=g)


class Discriminator(nn.Module):
    def __init__(self, img_size: int, channels: int = 3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


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

    dataSize = 128
    batchSize = 8
    imageSize = 64

    gpu = torch.device('cuda')

    catDataset = CatDataset(
        imageSubdirPath=r'E:\data\cat-vs-dog\cat',
        transform=transforms.Compose(
            [
                transforms.Resize((imageSize, imageSize)),
                # torchvision.transforms.functional.to_grayscale,
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: torch.reshape(x, x.shape[1:])),
                transforms.Normalize([0.5], [0.5])
            ]
        )
    )

    sampler = InfiniteSampler(catDataset)
    catLoader = DataLoader(catDataset, batch_size=batchSize, sampler=sampler)

    # Generate a random distance matrix.
    # # Make a matrix with positive values.
    # distancesCpu = np.clip(np.random.normal(0.5, 1.0 / 3, (dataSize, dataSize)), 0, 1)
    # # Make it symmetrical.
    # distancesCpu = np.matmul(distancesCpu, distancesCpu.T)

    # Generate random points and compute distances, guaranteeing that the triangle rule isn't broken.
    randomPoints = generate_points(dataSize)
    distancesCpu = scipy.spatial.distance_matrix(randomPoints, randomPoints, p=2)

    # catImagePath = os.path.expandvars(r'${DEV_METAPHOR_DATA_PATH}/cats/cat.247.jpg')
    # catImage = skimage.transform.resize(imageio.imread(catImagePath), (64, 64), 1).transpose(2, 0, 1)

    imagesInitCpu = np.clip(np.random.normal(0.5, 0.5 / 3, (dataSize, 3, imageSize, imageSize)), 0, 1)
    # imagesInitCpu = np.clip(np.tile(catImage, (dataSize, 1, 1, 1)) + np.random.normal(0., 0.5 / 6, (dataSize, 3, 64, 64)), 0, 1)
    images = torch.tensor(imagesInitCpu, requires_grad=True, dtype=torch.float32, device=gpu)

    scale = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=gpu)

    lossModel = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True).to(gpu)
    lossBce = torch.nn.BCELoss()

    discriminator = Discriminator(imageSize, 3).to(gpu)

    optimizerImages = torch.optim.Adam([images, scale], lr=1e-3, betas=(0.9, 0.999))
    optimizerDisc = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))

    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=batchSize // 2)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    catIter = iter(catLoader)
    for i in range(10000):

        realImageBatch, _ = next(catIter)  # type: Tuple(torch.Tensor, Any)
        realImageBatch = realImageBatch.to(gpu)
        # realImageBatch = torch.tensor(realImageBatchCpu, device=gpu)

        # noinspection PyTypeChecker
        randomIndices = np.random.randint(0, dataSize, batchSize).tolist()  # type: List[int]
        # randomIndices = list(range(dataSize))  # type: List[int]
        distanceBatch = torch.tensor(distancesCpu[randomIndices, :][:, randomIndices], dtype=torch.float32, device=gpu)
        imageBatch = images[randomIndices].contiguous()

        distPred = lossModel.forward(imageBatch.repeat(repeats=(batchSize, 1, 1, 1)).contiguous(),
                                     imageBatch.repeat_interleave(repeats=batchSize, dim=0).contiguous(), normalize=True)
        distPredMat = distPred.reshape((batchSize, batchSize))

        lossDist = torch.sum((distanceBatch - distPredMat * scale) ** 2)  # MSE
        lossRealness = lossBce(discriminator(imageBatch.detach()), torch.ones(imageBatch.shape[0], 1, device=gpu))
        lossImages = lossDist + 100.0 * lossRealness

        lossDiscReal = lossBce(discriminator(realImageBatch), torch.ones(realImageBatch.shape[0], 1, device=gpu))
        lossDiscFake = lossBce(discriminator(imageBatch.detach()), torch.zeros(imageBatch.shape[0], 1, device=gpu))
        lossDisc = (lossDiscFake + lossDiscReal) / 2

        optimizerImages.zero_grad()
        lossImages.backward()
        optimizerImages.step()

        optimizerDisc.zero_grad()
        lossDisc.backward()
        optimizerDisc.step()

        with torch.no_grad():
            # todo  We're clamping all the images every batch, can we do clamp only the ones updated?
            images = torch.clamp(images, 0, 1)

        if i % 100 == 0:
            msg = 'iter {}, loss images {:.3f}, loss dist {:.3f}, loss real {:.3f}, loss disc {:.3f}, scale: {:.3f}'.format(
                i, lossImages.item(), lossDist.item(), lossRealness.item(), lossDisc.item(), scale.item()
            )
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
