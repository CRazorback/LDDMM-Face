from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
from numpy.lib.stride_tricks import broadcast_arrays

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import HighResolutionNet, BasicBlock, Bottleneck
from .lddmm import *


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class LDDMMHighResolutionNet(HighResolutionNet):
    def __init__(self, config, stage=0, **kwargs):
        super(LDDMMHighResolutionNet, self).__init__(config)

        self.is_train = not config.TEST.INFERENCE
        self.points = config.MODEL.NUM_JOINTS if self.is_train else config.TEST.NUM_JOINTS
        self.stage = stage
        self.scale = config.DATASET.BOUNDINGBOX_SCALE_FACTOR
        self.train_fe = config.MODEL.FINETUNE_FE
        self.inplanes = 64
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # Regression Head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        self.regressor = nn.Linear(270, config.MODEL.NUM_JOINTS*2)
        
        self.init_landmarks = np.load('data/init_landmark.npy')
        self.index44 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26,
                        27, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                        48, 50, 52, 54, 55, 57, 59, 60, 62, 64, 65, 67]
        logger.info('=> loading initial landmarks...')
        self.init_landmarks = torch.Tensor(self.init_landmarks).cuda()
        self.init_landmarks = self.init_landmarks * config.MODEL.HEATMAP_SIZE[0] / 112
        
        if self.is_train:
            self.deform = LandmarkDeformLayer(n_landmark=config.MODEL.NUM_JOINTS)
        else:
            # self.deform = LandmarkDeformLayer2(train_init_landmark=self.init_landmarks[self.index44],
            #         train_n_landmark=config.MODEL.NUM_JOINTS, test_n_landmark=config.TEST.NUM_JOINTS)
            if config.TEST.NUM_JOINTS == config.MODEL.NUM_JOINTS:
                self.deform = LandmarkDeformLayer(n_landmark=config.TEST.NUM_JOINTS)
            else:
                self.deform = LandmarkDeformLayer(n_landmark=config.TEST.NUM_JOINTS, broadcast_index=self.index44)
                    
        if self.points == 44:
            self.init_landmarks = self.init_landmarks[self.index44]
        # 1.25 scale
        # if self.scale == 1.25:
        #     self.init_landmarks -= (config.MODEL.HEATMAP_SIZE[0] // 2)
        #     self.init_landmarks *= 1.25
        #     self.init_landmarks += (config.MODEL.HEATMAP_SIZE[0] // 2)

        self.landmark2img = LandmarkImageLayer(patch_size=6)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward_function(self, x, init_pts=None):
        # h, w = x.size(2), x.size(3)
        if init_pts is None:
            init_pts = torch.cat([self.init_landmarks.unsqueeze(0)]*x.size(0), dim=0)

        with torch.set_grad_enabled(self.train_fe):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.layer1(x)

            x_list = []
            for i in range(self.stage2_cfg['NUM_BRANCHES']):
                if self.transition1[i] is not None:
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(x)
            y_list = self.stage2(x_list)

            x_list = []
            for i in range(self.stage3_cfg['NUM_BRANCHES']):
                if self.transition2[i] is not None:
                    x_list.append(self.transition2[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage3(x_list)

            x_list = []
            for i in range(self.stage4_cfg['NUM_BRANCHES']):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage4(x_list)

        # Regression Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)
        # x = y_list
        # height, width = x[0].size(2), x[0].size(3)
        # x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
        # x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
        # x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
        # y = torch.cat([x[0], x1, x2, x3], dim=1)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                                [2:]).view(y.size(0), -1)

        alpha = self.regressor(y)
        deformed_pts = self.deform(alpha, init_pts)

        return deformed_pts

    def forward(self, x, init_pts=None):
        output = self.forward_function(x, init_pts)

        # # debug
        # import random
        # import imageio
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # filename = './debug/{}_{}.jpg'
        # index = random.randint(0, 100)
        # origin_img = x[1].detach().cpu().permute(1, 2, 0).numpy()
        # origin_img = origin_img * std + mean
        # # landmark image
        # landmark_imgs = self.landmark2img(x, output * 256 / 112)
        # # landmark40 = self.landmark2img(x[0].unsqueeze(0), self.deform.train_init_landmark.unsqueeze(0) * 256 / 112)
        # landmark_img = landmark_imgs[1, 0].detach().cpu().numpy()
        # # landmark40 = landmark40[0, 0].detach().cpu().numpy()
        # overlap = np.where(landmark_img == 0, origin_img[..., 0], landmark_img)
        # imageio.imwrite(filename.format('origin', index), (origin_img*255).astype(np.uint8))
        # imageio.imwrite(filename.format('overlap', index), (overlap*255).astype(np.uint8))
        # # imageio.imwrite(filename.format('init_landmark', index), (landmark40*255).astype(np.uint8))

        return output
