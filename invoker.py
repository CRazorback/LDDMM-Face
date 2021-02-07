import argparse
from lib.config import config, update_config
import sys
import numpy as np
from lib.datasets import WFLW, Face300W

# parser = argparse.ArgumentParser(description='Train Face Alignment')
# parser.add_argument('--cfg', help='experiment configuration filename',
#                     required=True, type=str)
# args = parser.parse_args()
# update_config(config, args)
# # dataset = WFLW(config, is_train=False)
# # landmarks = np.zeros([len(dataset), 98, 2])
# dataset = Face300W(config, is_train=False)
# landmarks = np.zeros([3149, 68, 2])
# for i, (a, b, c) in enumerate(dataset):
#     landmarks[i] = c['tpts']

# mean_shape = np.mean(landmarks, dim=0)
# print(mean_shape)

# np.save('./data/300w/init_landmark.npy', mean_shape.numpy())
curve2landmark = {
    0: np.arange(0, 9),
    1: np.arange(9, 17),
    2: np.arange(17, 22),
    3: np.arange(22, 27),
    4: np.arange(27, 31),
    5: np.arange(31, 36),
    6: np.arange(36, 42),
    7: np.arange(42, 48),
    8: np.arange(48, 55),
    9: np.arange(55, 62),
    10: np.arange(62, 67),
    11: np.arange(67, 72)}
init_landmarks = np.load('./data/init_landmark.npy')
index = list(np.arange(0, 48)) + list(np.arange(48, 55)) + list(np.arange(54, 60)) +\
                    [48] + list(np.arange(60, 65)) + list(np.arange(64, 68)) + [60]
init_landmarks = init_landmarks[index]
# curve2landmark = {
#     0: np.arange(0, 17),
#     1: np.arange(17, 33),
#     2: np.arange(33, 42),
#     3: np.arange(42, 51),
#     4: np.arange(51, 55),
#     5: np.arange(55, 60),
#     6: np.arange(60, 68),
#     7: np.arange(68, 76),
#     8: np.arange(76, 83),
#     9: np.arange(83, 88),
#     10: np.arange(88, 93),
#     11: np.arange(93, 96)
# }
# init_landmarks = np.load('data/wflw/init_landmark.npy')

sigmaV = np.zeros([12])
sigmaW = np.zeros([12])
for curve_idx, landmark_idxs in curve2landmark.items():
    landmarks = init_landmarks[landmark_idxs]
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    max_x = np.max(landmarks[:, 0])
    min_x = np.min(landmarks[:, 0])
    max_y = np.max(landmarks[:, 1])
    min_y = np.min(landmarks[:, 1])
    sigmaV[curve_idx] += np.mean([max_x - min_x, max_y - min_y])

print(sigmaV)