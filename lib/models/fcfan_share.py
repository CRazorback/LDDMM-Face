from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import math
import random
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lddmm_hrnet import LDDMMHighResolutionNet
from .lddmm import *

class RFCFAN(nn.Module):
    def __init__(self, config, **kwargs):
        super(RFCFAN, self).__init__()
        self.config = config
        
        self.stage1 = LDDMMHighResolutionNet(config, stage=0, **kwargs)
        self.stage2 = LDDMMHighResolutionNet(config, stage=0, **kwargs)
        self.deform = LandmarkDeformLayer()
        self.img_affine = AffineTransformLayer()
        self.landmark_affine = LandmarkTransformLayer()
        self.estimate = TransformParamsLayer()
        self.init_landmarks = np.load('data/meanFaceShape.npz')['meanShape'] * 0.8
        self.init_landmarks += 80
        self.init_landmarks = self.init_landmarks * config.MODEL.HEATMAP_SIZE[0] / 160
        self.init_landmarks = torch.Tensor(self.init_landmarks).cuda()
        self.proxy = nn.Linear(136, 136)
        # freeze stage 1 & 2
        self.stage1.requires_grad_(False)
        self.stage2.requires_grad_(False)
        self.landmark2img = LandmarkImageLayer(patch_size=8)

    def forward(self, x):
        with torch.no_grad():
            mean_shape = torch.cat([self.init_landmarks.unsqueeze(0)]*x.size(0), dim=0)
            pred1 = self.stage1(x)
            params = self.estimate(pred1, mean_shape)
            deformed_pred1 = self.landmark_affine(pred1, params)
            deformed_x = self.img_affine(x, params)
            pred2 = self.stage1(deformed_x)
            output = self.landmark_affine(pred2, params, inverse=True)
            
        # pred2 = self.proxy(pred2)
        self.debug(x, deformed_x, deformed_pred1, pred2)

        return output.view(-1, 68, 2)

    def debug(self, x, deformed_x, affined_preds, pred2):
        # debug
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        filename = './debug/{}_{}.jpg'
        index = random.randint(0, 100)
        deformed_img = deformed_x[0].detach().cpu().permute(1, 2, 0).numpy()
        deformed_img = deformed_img * std + mean
        origin_img = x[0].detach().cpu().permute(1, 2, 0).numpy()
        origin_img = origin_img * std + mean
        # landmark image
        landmark_imgs1 = self.landmark2img(x, affined_preds)
        landmark_imgs2 = self.landmark2img(x, pred2)
        landmark_img1 = landmark_imgs1[0, 0].detach().cpu().numpy()
        landmark_img2 = landmark_imgs2[0, 0].detach().cpu().numpy()
        overlap1 = np.where(landmark_img1 == 0, deformed_img[..., 0], landmark_img1)
        overlap2 = np.where(landmark_img2 == 0, deformed_img[..., 0], landmark_img2)
        imageio.imwrite(filename.format('origin', index), (origin_img*255).astype(np.uint8))
        imageio.imwrite(filename.format('overlap_pred1', index), (overlap1*255).astype(np.uint8))
        imageio.imwrite(filename.format('overlap_pred2', index), (overlap2*255).astype(np.uint8))

