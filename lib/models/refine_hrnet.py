from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import HighResolutionNet
from .lddmm_hrnet import LDDMMHighResolutionNet
from .lddmm import *
from ..core.evaluation import get_preds

class RefineHighResolutionNet(nn.Module):
    def __init__(self, config, **kwargs):
        super(RefineHighResolutionNet, self).__init__()
        self.config = config
        
        self.hrnet = HighResolutionNet(config, **kwargs)
        self.lddmm_hrnet = LDDMMHighResolutionNet(config, **kwargs)
        self.img_affine = AffineTransformLayer()
        self.landmark_affine = LandmarkTransformLayer()
        self.estimate = TransformParamsLayer()

    def forward(self, x):
        self.hrnet.eval()
        score_map = self.hrnet(x)
        res = [64, 64]

        coords = get_preds(score_map)  # float type
        
        # post-processing
        for n in range(coords.size(0)):
            for p in range(coords.size(1)):
                hm = score_map[n][p]
                px = int(math.floor(coords[n][p][0]))
                py = int(math.floor(coords[n][p][1]))
                if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                    diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]]).cuda()
                    coords[n][p] += diff.sign() * .25
        coords += 0.5
        preds = coords.clone()
        preds = preds * 112 / 64

        return preds
        # affine
        mean_shape = torch.cat([self.lddmm_hrnet.init_landmarks.unsqueeze(0)]*x.size(0), dim=0)
        # transform mean face
        mean_shape -= 56
        mean_shape = mean_shape * 1.25 + 56 
        params = self.estimate(preds, mean_shape)
        deformed_x = self.img_affine(x, params)
        affined_preds = self.landmark_affine(preds, params)
        deformed_preds = self.lddmm_hrnet(deformed_x, affined_preds)
        final_preds = self.landmark_affine(deformed_preds, params)

        # return final_preds.view(-1, 68, 2)

