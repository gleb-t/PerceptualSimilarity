
from __future__ import absolute_import

import sys
import scipy
import imageio
import scipy.misc
import numpy as np
import torch
from torch.autograd import Variable
import models

from typing import *

use_gpu = True

ref_path  = './imgs/ex_ref.png'
pred_path = './imgs/ex_p1.png'

ref_img = imageio.imread(ref_path).transpose(2, 0, 1) / 255.
pred_img = imageio.imread(pred_path).transpose(2, 0, 1) / 255.

gpu = torch.device('cuda')

# Torchify
# ref = torch.tensor(ref_img[np.newaxis, ...]).to(gpu)
# pred = torch.tensor(pred_img[np.newaxis, ...], requires_grad=True).to(gpu)

# Generate a random distance matrix.
dataSize = 8
# Make a matrix with positive values.
distancesCpu = np.clip(np.random.normal(0.5, 1.0 / 3, (dataSize, dataSize)), 0, 1)
# Make it symmetrical.
distancesCpu = np.matmul(distancesCpu, distancesCpu.T)

imagesCpu = np.clip(np.random.normal(0.5, 0.5 / 3, (dataSize, 3, 64, 64)), 0, 1)
images = torch.tensor(imagesCpu, requires_grad=True, dtype=torch.float32, device=gpu)

scale = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=gpu)


lossModel = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=use_gpu).to(gpu)
optimizer = torch.optim.Adam([images, scale], lr=1e-3, betas=(0.9, 0.999))
bs = 8

import matplotlib.pyplot as plt
plt.ion()
fig, axes = plt.subplots(nrows=2, ncols=bs // 2)

for i in range(10000):

    # randomIndices = np.random.randint(0, dataSize, bs).tolist()  # type: List[int]
    randomIndices = list(range(dataSize))  # type: List[int]
    distanceBatch = torch.tensor(distancesCpu[randomIndices, :][:, randomIndices], dtype=torch.float32, device=gpu)
    imageBatch = images[randomIndices].contiguous()

    distPred = lossModel.forward(imageBatch.repeat(repeats=(bs, 1, 1, 1)).contiguous(),
                                 imageBatch.repeat_interleave(repeats=bs, dim=0).contiguous(), normalize=True)
    distPredMat = distPred.reshape((bs, bs))

    loss = torch.sum((distanceBatch - distPredMat * scale) ** 2)  # MSE

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    images.data = torch.clamp(images.data, 0, 1)
    
    if i % 10 == 0:
        msg = 'iter {}, loss {:.3f}, scale: {:.3f}'.format(i, loss.item(), scale.item())
        print(msg)
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(imageBatch.cpu().data.numpy()[i].transpose(1, 2, 0))
        fig.suptitle(msg)
        plt.pause(5e-2)
        # plt.imsave('imgs_saved/%04d.jpg'%i,pred_img)


