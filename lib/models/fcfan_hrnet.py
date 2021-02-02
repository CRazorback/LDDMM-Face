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

class FCFAN(nn.Module):
    def __init__(self, config, **kwargs):
        super(FCFAN, self).__init__()
        self.config = config
        
        self.stage1 = LDDMMHighResolutionNet(config, stage=1, **kwargs)
        self.stage2 = LDDMMHighResolutionNet(config, stage=2, **kwargs)
        self.deform = LandmarkDeformLayer()
        self.img_affine = AffineTransformLayer()
        self.landmark_affine = LandmarkTransformLayer()
        self.estimate = TransformParamsLayer()
        self.landmark2img = LandmarkImageLayer(patch_size=8)
        self.init_landmarks = np.load('data/meanFaceShape.npz')['meanShape']   
        self.init_landmarks += 80
        self.init_landmarks = self.init_landmarks * config.MODEL.HEATMAP_SIZE[0] / 160
        self.init_landmarks = torch.Tensor(self.init_landmarks).cuda()
        # freeze stage 1
        # self.stage1.requires_grad_(False)

    def forward(self, x):
        mean_shape = torch.cat([self.init_landmarks.unsqueeze(0)]*x.size(0), dim=0)
        pred1, fe1 = self.stage1(x)
        params = self.estimate(pred1, mean_shape)
        deformed_x = self.img_affine(x, params)
        affined_preds = self.landmark_affine(pred1, params)
        # 1/4 heatmap size
        x_downsample = F.interpolate(x, size=[64, 64], mode='bilinear', align_corners=False)
        heatmap = self.landmark2img(x_downsample, affined_preds * 64 / self.config.MODEL.HEATMAP_SIZE[0])
        deformed_preds, _ = self.stage2(deformed_x, affined_preds, fe1, heatmap)
        pred2 = self.landmark_affine(deformed_preds, params, inverse=True)

        # return [pred1.view(-1, 68, 2), pred2.view(-1, 68, 2)]
        # return pred2.view(-1, 68, 2)
        return self.debug(x, deformed_x, affined_preds, pred1)

    def debug(self, x, deformed_x, affined_preds, pred1):
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
        landmark_imgs = self.landmark2img(x, affined_preds)
        landmark_img = landmark_imgs[0, 0].detach().cpu().numpy()
        overlap = np.where(landmark_img == 0, deformed_img[..., 0], landmark_img)
        imageio.imwrite(filename.format('deformed', index), (deformed_img*255).astype(np.uint8))
        imageio.imwrite(filename.format('origin', index), (origin_img*255).astype(np.uint8))
        imageio.imwrite(filename.format('overlap', index), (overlap*255).astype(np.uint8))

        return pred1.view(-1, 68, 2)

