from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lddmm import *
from .hourglass import HourglassNet

logger = logging.getLogger(__name__)


class LDDMM_Hourglass(HourglassNet):
    def __init__(self, config):
        super(LDDMM_Hourglass, self).__init__(config)

        self.is_train = not config.TEST.INFERENCE
        self.points = config.MODEL.NUM_JOINTS if self.is_train else config.TEST.NUM_JOINTS
        self.scale = config.DATASET.BOUNDINGBOX_SCALE_FACTOR
        self.train_fe = config.MODEL.FINETUNE_FE

        # regression head
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.regressor = nn.Linear(2048, config.MODEL.NUM_JOINTS*2)

        logger.info('=> loading initial landmarks...')
        self.init_landmarks = torch.Tensor(self.init_landmarks).cuda()
        self.init_landmarks = self.init_landmarks * config.MODEL.HEATMAP_SIZE[0] / 112
        
        if self.is_train:
            self.deform = LandmarkDeformLayer(config, n_landmark=config.MODEL.NUM_JOINTS)
        else:
            self.deform = LandmarkDeformLayer(config, n_landmark=config.TEST.NUM_JOINTS)
                    
        if self.points != 68:
            self.index = get_index(config.DATASET.DATASET, self.points)
            self.init_landmarks = self.init_landmarks[self.index]

    def _make_conv_with_stride(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(outplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=2, padding=1, bias=False)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x, init_pts=None):
        if init_pts is None:
            init_pts = torch.cat([self.init_landmarks.unsqueeze(0)]*x.size(0), dim=0)

        with torch.set_grad_enabled(self.train_fe):
            fe = []
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.maxpool(x)
            x = self.layer2(x)
            x = self.layer3(x)

            for i in range(self.num_stacks):
                y = self.hg[i](x)
                y = self.res[i](y)
                y = self.fc[i](y)
                score = self.score[i](y)
                fe.append(y)
                if i < self.num_stacks-1:
                    fc_ = self.fc_[i](y)
                    score_ = self.score_[i](score)
                    x = x + fc_ + score_

        fe_cat = torch.cat(fe, dim=1)
        y = self.final_layer(fe_cat)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                                [2:]).view(y.size(0), -1)

        alpha = self.regressor(y)
        deformed_pts = self.deform(alpha, init_pts)

        return deformed_pts

    def init_weights(self, pretrained=''):
        logger.info('=> init weights in default')
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            if 'lddmm' in pretrained or 'best' in pretrained:
                pretrained_dict = pretrained_dict.state_dict()
            if 'checkpoint' in pretrained:
                pretrained_dict = pretrained_dict['state_dict'].state_dict()
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            available_pretrained_dict = {}

            for k, v in pretrained_dict.items():
                if k in model_dict.keys():
                    available_pretrained_dict[k] = v
                if k[7:] in model_dict.keys():
                    available_pretrained_dict[k[7:]] = v

            for k, _ in available_pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(available_pretrained_dict)
            self.load_state_dict(model_dict)

