import datetime
import itertools
import math
import os
import random
import glob
import json
from typing import *

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
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import skimage.transform
import matplotlib.pyplot as plt


import elpips
from PythonExtras.distance_matrix import render_distance_matrix, DistanceMatrixConfig


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



def l2_sqr_dist_matrix(x: torch.Tensor) -> torch.Tensor:
    gram = torch.mm(x, x.T)
    diag = gram.diagonal()

    return diag.unsqueeze(1) + diag.unsqueeze(0) - 2 * gram


def main():

    dataSize = 32
    batchSize = 8
    elpipsBatchSize = 1
    # imageSize = 32
    imageSize = 64
    nz = 100

    # discCheckpointPath = r'E:\projects\visus\PyTorch-GAN\implementations\dcgan\checkpoints\2020_07_10_15_53_34\disc_step4800.pth'
    discCheckpointPath = r'E:\projects\visus\pytorch-examples\dcgan\out\netD_epoch_24.pth'
    genCheckpointPath = r'E:\projects\visus\pytorch-examples\dcgan\out\netG_epoch_24.pth'

    gpu = torch.device('cuda')

    # For now we normalize the vectors to have norm 1, but don't make sure
    # that the data has certain mean/std.
    pointDataset = AuthorDataset(
        jsonPath=r'E:\out\scripts\metaphor-vis\authors-all.json'
    )

    # Take top N points.
    points = np.asarray([pointDataset[i][0] for i in range(dataSize)])
    distPointsCpu = l2_sqr_dist_matrix(torch.tensor(points)).numpy()

    latents = torch.tensor(np.random.normal(0.0, 1.0, (dataSize, nz)),
                           requires_grad=True, dtype=torch.float32, device=gpu)

    scale = torch.tensor(2.7, requires_grad=True, dtype=torch.float32, device=gpu)  # todo Re-check!
    bias = torch.tensor(0.0, requires_grad=True, dtype=torch.float32, device=gpu)  # todo Re-check!

    lpips = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True).to(gpu)
    # lossModel = lpips
    config = elpips.Config()
    config.batch_size = elpipsBatchSize  # Ensemble size for ELPIPS.
    config.set_scale_levels_by_image_size(imageSize, imageSize)
    lossModel = elpips.ElpipsMetric(config, lpips).to(gpu)

    discriminator = Discriminator(3, 64, 1)
    if discCheckpointPath:
        discriminator.load_state_dict(torch.load(discCheckpointPath))
    else:
        discriminator.init_params()
    discriminator = discriminator.to(gpu)

    generator = Generator(nz=nz, ngf=64)
    if genCheckpointPath:
        generator.load_state_dict(torch.load(genCheckpointPath))
    else:
        generator.init_params()
    generator = generator.to(gpu)

    # optimizerImages = torch.optim.Adam([images, scale], lr=1e-2, betas=(0.9, 0.999))
    optimizerScale = torch.optim.Adam([scale, bias], lr=0.001)
    # optimizerGen = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizerDisc = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.9, 0.999))
    # optimizerDisc = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerLatents = torch.optim.Adam([latents], lr=5e-3, betas=(0.9, 0.999))

    fig, axes = plt.subplots(nrows=2, ncols=batchSize // 2)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    outPath = os.path.join('runs', datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(outPath)

    summaryWriter = SummaryWriter(outPath)

    for batchIndex in range(10000):

        # noinspection PyTypeChecker
        randomIndices = np.random.randint(0, dataSize, batchSize).tolist()  # type: List[int]
        # # randomIndices = list(range(dataSize))  # type: List[int]
        distTarget = torch.tensor(distPointsCpu[randomIndices, :][:, randomIndices], dtype=torch.float32, device=gpu)
        latentsBatch = latents[randomIndices]

        imageBatchFake = generator(latentsBatch[:, :, None, None].float())

        # todo It's possible to compute this more efficiently, but would require re-implementing lpips.
        # For now, compute the full BSxBS matrix row-by-row to avoid memory issues.
        lossDistTotal = torch.tensor(0.0, device=gpu)
        distanceRows = []
        for iRow in range(batchSize):
            distPredFlat = lossModel(imageBatchFake[iRow].repeat(repeats=(batchSize, 1, 1, 1)).contiguous(),
                                     imageBatchFake, normalize=True)
            distPred = distPredFlat.reshape((1, batchSize))
            distanceRows.append(distPred)
            lossDist = torch.sum((distTarget[iRow] - (distPred * scale + bias)) ** 2)  # MSE
            lossDistTotal += lossDist

        lossDistTotal /= batchSize * batchSize  # Compute the mean.

        distPredFull = torch.cat(distanceRows, dim=0)

        # print('{} - {} || {} - {}'.format(
        #     torch.min(distPred).item(),
        #     torch.max(distPred).item(),
        #     torch.min(distTarget).item(),
        #     torch.max(distTarget).item()
        # ))

        # discPred = discriminator(imageBatchFake)
        # lossRealness = bceLoss(discPred, torch.ones(imageBatchFake.shape[0], device=gpu))
        # lossGen = lossDist + 1.0 * lossRealness
        lossLatents = lossDistTotal

        # optimizerGen.zero_grad()
        # optimizerScale.zero_grad()
        # lossGen.backward()
        # optimizerGen.step()
        # optimizerScale.step()

        optimizerLatents.zero_grad()
        # optimizerScale.zero_grad()
        lossLatents.backward()
        optimizerLatents.step()
        # optimizerScale.step()

        # with torch.no_grad():
        #     # todo  We're clamping all the images every batch, can we clamp only the ones updated?
        #     # images = torch.clamp(images, 0, 1)  # For some reason this was making the training worse.
        #     images.data = torch.clamp(images.data, 0, 1)

        if batchIndex % 100 == 0:
            msg = 'iter {} loss dist {:.3f} scale: {:.3f} bias: {:.3f}'.format(batchIndex, lossDistTotal.item(), scale.item(), bias.item())
            print(msg)

            summaryWriter.add_scalar('loss-dist', lossDistTotal.item(), global_step=batchIndex)

            def gpu_images_to_numpy(images):
                imagesNumpy = images.cpu().data.numpy().transpose(0, 2, 3, 1)
                imagesNumpy = (imagesNumpy + 1) / 2

                return imagesNumpy

            # print(discPred.tolist())
            imageBatchFakeCpu = gpu_images_to_numpy(imageBatchFake)
            # imageBatchRealCpu = gpu_images_to_numpy(imageBatchReal)
            for iCol, ax in enumerate(axes.flatten()[:batchSize]):
                ax.imshow(imageBatchFakeCpu[iCol])
            fig.suptitle(msg)

            with torch.no_grad():
                images = gpu_images_to_numpy(generator(latents[..., None, None]))

                authorVectorsProj = umap.UMAP(n_neighbors=min(5, dataSize), random_state=1337).fit_transform(points)
                plot_image_scatter(ax2, authorVectorsProj, images, downscaleRatio=2)

            fig.savefig(os.path.join(outPath, f'batch_{batchIndex}.png'))
            fig2.savefig(os.path.join(outPath, f'scatter_{batchIndex}.png'))
            plt.close(fig)
            plt.close(fig2)

            with torch.no_grad():
                imagesGpu = generator(latents[..., None, None])
                imageNumber = imagesGpu.shape[0]

                # Compute LPIPS distances, batch to avoid memory issues.
                bs = min(imageNumber, 8)
                assert imageNumber % bs == 0
                distPredEval = np.zeros((imagesGpu.shape[0], imagesGpu.shape[0]))
                for iCol in range(imageNumber // bs):
                    startA, endA = iCol * bs, (iCol + 1) * bs
                    imagesA = imagesGpu[startA:endA]
                    for j in range(imageNumber // bs):
                        startB, endB = j * bs, (j + 1) * bs
                        imagesB = imagesGpu[startB:endB]

                        distBatchEval = lossModel(imagesA.repeat(repeats=(bs, 1, 1, 1)).contiguous(),
                                                  imagesB.repeat_interleave(repeats=bs, dim=0).contiguous(),
                                                  normalize=True).cpu().numpy()

                        distPredEval[startA:endA, startB:endB] = distBatchEval.reshape((bs, bs))

                distPredEval = (distPredEval * scale.item() + bias.item())

                # Move to the CPU and append an alpha channel for rendering.
                images = gpu_images_to_numpy(imagesGpu)
                images = [np.concatenate([im, np.ones(im.shape[:-1] + (1,))], axis=-1) for im in images]

                distPoints = distPointsCpu
                assert np.abs(distPoints - distPoints.T).max() < 1e-5
                distPoints = np.minimum(distPoints, distPoints.T)  # Remove rounding errors, guarantee symmetry.
                config = DistanceMatrixConfig()
                config.dataRange = (0., 4.)
                _, pointIndicesSorted = render_distance_matrix(
                    os.path.join(outPath, f'dist_point_{batchIndex}.png'),
                    distPoints,
                    images,
                    config=config
                )

                # print(np.abs(distPredFlat - distPredFlat.T).max())
                # assert np.abs(distPredFlat - distPredFlat.T).max() < 1e-5
                # todo The symmetry doesn't hold for E-LPIPS, since it's stochastic.
                distPredEval = np.minimum(distPredEval, distPredEval.T)  # Remove rounding errors, guarantee symmetry.
                config = DistanceMatrixConfig()
                config.dataRange = (0., 4.)
                render_distance_matrix(
                    os.path.join(outPath, f'dist_images_{batchIndex}.png'),
                    distPredEval,
                    images,
                    config=config
                )

                config = DistanceMatrixConfig()
                config.dataRange = (0., 4.)
                render_distance_matrix(
                    os.path.join(outPath, f'dist_images_aligned_{batchIndex}.png'),
                    distPredEval,
                    images,
                    predefinedOrder=pointIndicesSorted,
                    config=config
                )

                fig, axes = plt.subplots(ncols=2)
                axes[0].matshow(distTarget.cpu().numpy(), vmin=0, vmax=4)
                axes[1].matshow(distPredFull.cpu().numpy() * scale.item(), vmin=0, vmax=4)
                fig.savefig(os.path.join(outPath, f'batch_dist_{batchIndex}.png'))
                plt.close(fig)

                surveySize = 30
                fig, axes = plt.subplots(nrows=3, ncols=surveySize, figsize=(surveySize, 3))
                assert len(images) == dataSize
                allIndices = list(range(dataSize))
                with open(os.path.join(outPath, f'survey_{batchIndex}.txt'), 'w') as file:
                    for iCol in range(surveySize):
                        randomIndices = random.sample(allIndices, k=3)
                        leftToMid = distPointsCpu[randomIndices[0], randomIndices[1]]
                        rightToMid = distPointsCpu[randomIndices[2], randomIndices[1]]

                        correctAnswer = 'left' if leftToMid < rightToMid else 'right'
                        file.write("{}\t{}\t{}\t{}\t{}\n".format(iCol, correctAnswer, leftToMid, rightToMid,
                                                                 str(tuple(randomIndices))))

                        for iRow in (0, 1, 2):
                            axes[iRow][iCol].imshow(images[randomIndices[iRow]])

                fig.savefig(os.path.join(outPath, f'survey_{batchIndex}.png'))
                plt.close(fig)

            torch.save(generator.state_dict(), os.path.join(outPath, 'gen_{}.pth'.format(batchIndex)))
            torch.save(discriminator.state_dict(), os.path.join(outPath, 'gen_{}.pth'.format(batchIndex)))

    summaryWriter.close()


if __name__ == '__main__':
    main()
