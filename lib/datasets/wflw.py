# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
from lib.utils.lddmm_params import get_index


class WFLW(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.dataset_name = 'WFLW'
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.points = cfg.MODEL.NUM_JOINTS if is_train else cfg.TEST.NUM_JOINTS
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.bounding_box_scale_factor = cfg.DATASET.BOUNDINGBOX_SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)
        self.index = get_index('WFLW', self.points)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='WFLW')
                center[0] = img.shape[1] - center[0]

        img = crop(img, center, scale, self.input_size, rot=r)
        if self.label_type == 'Gaussian':
            target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=r)
                if self.label_type == 'Gaussian':                               
                    target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                                label_type=self.label_type)

        # import imageio
        # import cv2
        # img1, img2 = img.copy(), img.copy()
        # tpts = tpts.astype(np.uint8)
        # for k in range(tpts.shape[0]):
        #     cv2.circle(img1, (tpts[k, 0], tpts[k, 1]), 3, [0, 255, 0], -1)
        # imageio.imwrite('Figure1-WFLW/full{}.jpg'.format(idx), img1)
        # tpts_ = tpts[self.index]
        # for k in range(tpts_.shape[0]):
        #     cv2.circle(img2, (tpts_[k, 0], tpts_[k, 1]), 3, [0, 255, 0], -1)
        # imageio.imwrite('Figure1-WFLW/weak{}.jpg'.format(idx), img2)
        # print(idx)

        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        if self.label_type == 'Gaussian':
            target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        origin_pts = torch.Tensor(pts)

        # weak-supervised
        if self.points < 98:
            tpts = tpts[self.index]
            pts = pts[self.index]
            origin_pts = origin_pts[self.index]
            if self.label_type == 'Gaussian':
                target = target[self.index]

        meta = {'index': idx, 'center': center, 'scale': scale, 'dataset_name': self.dataset_name,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'origin_pts': origin_pts}

        if self.label_type == 'Gaussian':
            return img, target, meta
        else:
            return img, tpts, meta


if __name__ == '__main__':
    pass
