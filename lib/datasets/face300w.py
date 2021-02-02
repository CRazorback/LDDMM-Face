# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random
import itertools
import imageio

import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from lib.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class Face300W(data.Dataset):

    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.points = cfg.MODEL.NUM_JOINTS
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.bounding_box_scale_factor = cfg.DATASET.BOUNDINGBOX_SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.curve2landmark = {
            0: np.arange(0, 9),
            1: np.arange(9, 17),
            2: np.arange(17, 22),
            3: np.arange(22, 27),
            4: np.arange(27, 31),
            5: np.arange(31, 36),
            6: np.arange(36, 42),
            7: np.arange(42, 48),
            8: np.arange(48, 55),
            9: np.arange(55, 60),
            10: np.arange(60, 65),
            11: np.arange(65, 68)}

        self.index44 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 21, 22, 24, 26,
                        27, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                        48, 50, 52, 54, 55, 57, 59, 60, 62, 64, 65, 67]

    def __len__(self):
        if self.is_train:
            return len(self.landmarks_frame) * 1
        else:
            return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # idx = idx % len(self.landmarks_frame)
        
        image_path = os.path.join(self.data_root,
                                  self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]

        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)

        # DAN scale or HRNet scale
        scale *= self.bounding_box_scale_factor
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
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
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
        # debug
        # filename = './debug/{}_{}.jpg'
        # index = random.randint(0, 100)
        # imageio.imwrite(filename.format('origin', index), img.astype(np.uint8))

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        if self.label_type == 'Gaussian':
            target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        origin_pts = torch.Tensor(pts)

        # weak-supervised
        if self.points == 44:
            tpts = tpts[self.index44]
            pts = pts[self.index44]
            if self.label_type == 'Gaussian':
                target = target[self.index44]

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'origin_pts': origin_pts}

        if self.label_type == 'Gaussian':
            return img, target, meta
        else:
            return img, tpts, meta


if __name__ == '__main__':
    import argparse
    from lib.config import config, update_config
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    args = parser.parse_args()
    update_config(config, args)
    config.DATASET.TESTSET = 'data/300w/face_landmarks_300w_train.csv'
    config.MODEL.NUM_JOINTS = 68
    config.MODEL.HEATMAP_SIZE = 112
    config.MODEL.TARGET_TYPE = 'Landmark'
    dataset = Face300W(config, is_train=False)
    landmarks = torch.zeros([len(dataset), 68, 2])
    sigmaV = torch.zeros([12])
    sigmaW = torch.zeros([12])
    for i, (a, b, c) in enumerate(dataset):
        for curve_idx, landmark_idxs in dataset.curve2landmark.items():
            sigmaV[curve_idx] += torch.max(b[landmark_idxs, 0]) - torch.min(b[landmark_idxs, 0])
            sigmaW[curve_idx] += torch.max(b[landmark_idxs, 1]) - torch.min(b[landmark_idxs, 1])
    