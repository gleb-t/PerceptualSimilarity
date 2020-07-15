import datetime
import itertools
import math
import os
import random
import glob
import json
import scipy
import imageio
import scipy.misc
import scipy.spatial
import scipy.ndimage
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import umap
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import skimage.transform

from typing import *


import models
from dcgan import Discriminator, Generator


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


class AuthorDataset(Dataset):

    def __init__(self, jsonPath: str, normalize=True):
        self.jsonPath = jsonPath

        with open(self.jsonPath, 'r') as file:
            self.data = json.load(file)

        self.names = self.data['names']
        self.vectors = np.asarray(self.data['vectors'])
        # Normalize the vectors.
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=-1, keepdims=True)

    def __getitem__(self, index):
        return self.vectors[index], self.names[index]

    def __len__(self):
        return len(self.names)


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


def l2_sqr_dist_matrix(x: torch.Tensor) -> torch.Tensor:
    gram = torch.mm(x, x.T)
    diag = gram.diagonal()

    return diag.unsqueeze(1) + diag.unsqueeze(0) - 2 * gram


def main():

    dataSize = 128
    batchSize = 8
    # imageSize = 32
    imageSize = 64

    # discCheckpointPath = r'E:\projects\visus\PyTorch-GAN\implementations\dcgan\checkpoints\2020_07_10_15_53_34\disc_step4800.pth'
    # discCheckpointPath = r'E:\projects\visus\pytorch-examples\dcgan\out\netD_epoch_24.pth'
    discCheckpointPath = None

    gpu = torch.device('cuda')

    # imageDataset = CatDataset(
    #     imageSubdirPath=r'E:\data\cat-vs-dog\cat',
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize((imageSize, imageSize)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.5], [0.5])
    #         ]
    #     )
    # )

    imageDataset = datasets.CIFAR10(root=r'e:\data\images\cifar10', download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize((imageSize, imageSize)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        transforms.Normalize([0.5], [0.5]),
                               ]))

    # For now we normalize the vectors to have norm 1, but don't make sure
    # that the data has certain mean/std.
    pointDataset = AuthorDataset(
        jsonPath=r'E:\out\scripts\metaphor-vis\authors-all.json'
    )

    imageLoader = DataLoader(imageDataset, batch_size=batchSize, sampler=InfiniteSampler(imageDataset))
    pointLoader = DataLoader(pointDataset, batch_size=batchSize, sampler=InfiniteSampler(pointDataset))

    # Generate a random distance matrix.
    # # Make a matrix with positive values.
    # distancesCpu = np.clip(np.random.normal(0.5, 1.0 / 3, (dataSize, dataSize)), 0, 1)
    # # Make it symmetrical.
    # distancesCpu = np.matmul(distancesCpu, distancesCpu.T)

    # Generate random points and compute distances, guaranteeing that the triangle rule isn't broken.
    # randomPoints = generate_points(dataSize)
    # distancesCpu = scipy.spatial.distance_matrix(randomPoints, randomPoints, p=2)


    # catImagePath = os.path.expandvars(r'${DEV_METAPHOR_DATA_PATH}/cats/cat.247.jpg')
    # catImage = skimage.transform.resize(imageio.imread(catImagePath), (64, 64), 1).transpose(2, 0, 1)

    # imagesInitCpu = np.clip(np.random.normal(0.5, 0.5 / 3, (dataSize, 3, imageSize, imageSize)), 0, 1)
    # imagesInitCpu = np.clip(np.tile(catImage, (dataSize, 1, 1, 1)) + np.random.normal(0., 0.5 / 6, (dataSize, 3, 64, 64)), 0, 1)
    # images = torch.tensor(imagesInitCpu, requires_grad=True, dtype=torch.float32, device=gpu)

    scale = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=gpu)

    lossModel = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True).to(gpu)
    bceLoss = torch.nn.BCELoss()

    # discriminator = Discriminator(imageSize, 3)
    discriminator = Discriminator(3, 64, 1)
    if discCheckpointPath:
        discriminator.load_state_dict(torch.load(discCheckpointPath))
    else:
        discriminator.init_params()

    discriminator = discriminator.to(gpu)

    generator = Generator(nz=pointDataset[0][0].shape[0], ngf=64)
    generator.init_params()
    generator = generator.to(gpu)

    # todo init properly, if training
    # discriminator.apply(weights_init_normal)

    # optimizerImages = torch.optim.Adam([images, scale], lr=1e-2, betas=(0.9, 0.999))
    optimizerScale = torch.optim.Adam([scale], lr=0.001)
    optimizerGen = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizerDisc = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))
    optimizerDisc = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=2 * 2, ncols=batchSize // 2)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    outPath = os.path.join('images', datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(outPath)

    imageIter = iter(imageLoader)
    pointIter = iter(pointLoader)
    for batchIndex in range(10000):

        imageBatchReal, _ = next(imageIter)  # type: Tuple(torch.Tensor, Any)
        imageBatchReal = imageBatchReal.to(gpu)
        # imageBatchReal = torch.tensor(realImageBatchCpu, device=gpu)

        # noinspection PyTypeChecker
        # randomIndices = np.random.randint(0, dataSize, batchSize).tolist()  # type: List[int]
        # # randomIndices = list(range(dataSize))  # type: List[int]
        # distanceBatch = torch.tensor(distancesCpu[randomIndices, :][:, randomIndices], dtype=torch.float32, device=gpu)
        # imageBatchFake = images[randomIndices].contiguous()
        vectorBatch, _ = next(pointIter)
        vectorBatch = vectorBatch.to(gpu)
        distanceBatch = l2_sqr_dist_matrix(vectorBatch)  # In-batch vector distances.

        imageBatchFake = generator(vectorBatch[:, :, None, None].float())

        # todo It's possible to compute this more efficiently, but would require re-implementing lpips.
        distPred = lossModel.forward(imageBatchFake.repeat(repeats=(batchSize, 1, 1, 1)).contiguous(),
                                     imageBatchFake.repeat_interleave(repeats=batchSize, dim=0).contiguous(), normalize=True)
        distPredMat = distPred.reshape((batchSize, batchSize))

        lossDist = torch.sum((distanceBatch - distPredMat * scale) ** 2)  # MSE
        discPred = discriminator(imageBatchFake)
        lossRealness = bceLoss(discPred, torch.ones(imageBatchFake.shape[0], device=gpu))
        lossGen = lossDist + 10.0 * lossRealness

        optimizerGen.zero_grad()
        optimizerScale.zero_grad()
        lossGen.backward()
        optimizerGen.step()
        optimizerScale.step()

        lossDiscReal = bceLoss(discriminator(imageBatchReal), torch.ones(imageBatchReal.shape[0], device=gpu))
        lossDiscFake = bceLoss(discriminator(imageBatchFake.detach()), torch.zeros(imageBatchFake.shape[0], device=gpu))
        lossDisc = (lossDiscFake + lossDiscReal) / 2
        # lossDisc = torch.tensor(0)

        optimizerDisc.zero_grad()
        lossDisc.backward()
        optimizerDisc.step()

        # with torch.no_grad():
        #     # todo  We're clamping all the images every batch, can we clamp only the ones updated?
        #     # images = torch.clamp(images, 0, 1)  # For some reason this was making the training worse.
        #     images.data = torch.clamp(images.data, 0, 1)

        if batchIndex % 100 == 0:
            msg = 'iter {}, loss gen {:.3f}, loss dist {:.3f}, loss real {:.3f}, loss disc {:.3f}, scale: {:.3f}'.format(
                batchIndex, lossGen.item(), lossDist.item(), lossRealness.item(), lossDisc.item(), scale.item()
            )
            print(msg)

            def gpu_images_to_numpy(images):
                imagesNumpy = images.cpu().data.numpy().transpose(0, 2, 3, 1)
                imagesNumpy = (imagesNumpy + 1) / 2

                return imagesNumpy

            # print(discPred.tolist())
            imageBatchFakeCpu = gpu_images_to_numpy(imageBatchFake)
            imageBatchRealCpu = gpu_images_to_numpy(imageBatchReal)
            for i, ax in enumerate(axes.flatten()[:batchSize]):
                ax.imshow(imageBatchFakeCpu[i])
            for i, ax in enumerate(axes.flatten()[batchSize:]):
                ax.imshow(imageBatchRealCpu[i])
            fig.suptitle(msg)

            with torch.no_grad():
                points = np.asarray([pointDataset[i][0] for i in range(200)], dtype=np.float32)
                images = generator(torch.tensor(points[..., None, None], device=gpu)).cpu().numpy().transpose(0, 2, 3, 1)

                authorVectorsProj = umap.UMAP(n_neighbors=5, random_state=1337).fit_transform(points)
                plot_image_scatter(ax2, authorVectorsProj, (images + 1) / 2, downscaleRatio=2)

            fig.savefig(os.path.join(outPath, 'batch_{}.png'.format(batchIndex)))
            fig2.savefig(os.path.join(outPath, 'scatter_{}.png'.format(batchIndex)))
            plt.close(fig)
            plt.close(fig2)


if __name__ == '__main__':
    main()
