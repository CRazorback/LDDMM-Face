# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math
import torch
import torch.linalg
import torch.nn as nn
import numpy as np

import shapely.geometry as geom

from sklearn.neighbors import NearestNeighbors

from ..utils.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, meta):
    
    targets = meta['pts']
    if preds.size(1) != targets.size(1):
        targets = meta['origin_pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
            # rmse[i] = np.sum(np.linalg.norm(pts_pred[0:17:2] - pts_gt[0:17:2], axis=1)) / (interocular * 9)
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 44:  # 300w weak supverised
            # interocular
            interocular = np.linalg.norm(pts_gt[20, ] - pts_gt[29, ])
            # rmse[i] = np.sum(np.linalg.norm(pts_pred[0:9] - pts_gt[0:9], axis=1)) / (interocular * 9)
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: N'xm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def compute_curve_dist(preds, meta):
    targets = meta['origin_pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = targets.shape[1]
    norm_dists = np.zeros(N)
    norm_dists5 = np.zeros([5, N])

    curve2landmark68 = {
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
        11: np.arange(65, 68)
    }
    curve2landmark44 = {
        0: np.arange(0, 5),
        1: np.arange(5, 9),
        2: np.arange(9, 12),
        3: np.arange(12, 15),
        4: np.arange(15, 17),
        5: np.arange(17, 20),
        6: np.arange(20, 26),
        7: np.arange(26, 32),
        8: np.arange(32, 36),
        9: np.arange(36, 39),
        10: np.arange(39, 42),
        11: np.arange(42, 44)
    }
    idx_5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10, 11]]

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 44:  # 300w weak supverised
            # interocular
            interocular = np.linalg.norm(pts_gt[20, ] - pts_gt[29, ])
        else:
            raise ValueError('Number of landmarks is wrong')

        for j, curve_idxs in enumerate(idx_5):
            gt_landmark_idxs = []
            pred_landmark_idxs = []
            for curve_idx in curve_idxs:
                gt_landmark_idxs.extend(list(curve2landmark68[curve_idx]))
                if preds.shape[1] == 68:
                    pred_landmark_idxs.extend(list(curve2landmark68[curve_idx]))
                elif preds.shape[1] == 44:
                    pred_landmark_idxs.extend(list(curve2landmark44[curve_idx]))

            sub_pts_gt = pts_gt[gt_landmark_idxs]
            sub_pts_pred = pts_pred[pred_landmark_idxs]
            dist, _ = nearest_neighbor(sub_pts_pred, sub_pts_gt)
            dist_inv, _ = nearest_neighbor(sub_pts_gt, sub_pts_pred)
            norm_dists5[j, i] = np.mean(np.concatenate([dist, dist_inv])) / interocular

        norm_dists[i] = np.mean(norm_dists5[:, i])

    return norm_dists, norm_dists5


def perpendicualr_dist(p1, p2, p):
    return np.linalg.norm(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)


def point2curve(pts_set1, pts_set2, curve2landmark, landmark2curve):
    dist = np.zeros(pts_set1.shape[0])
    dist_5 = [[] for _ in range(5)]
    idx_5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10, 11]]
    for idx, pt in enumerate(pts_set1):
        pt_curve = landmark2curve[idx]
        # find shortest distance to curve
        curve_pts_idx = curve2landmark[pt_curve]
        curve_pts = pts_set2[curve_pts_idx]
        line = geom.LineString(curve_pts)
        point = geom.Point(pt[0], pt[1])
        dist[idx] = point.distance(line)
    
    for landmark_idx, curve_idx in landmark2curve.items():
        for i, idx in enumerate(idx_5):
            if curve_idx in idx:
                dist_5[i].append(dist[landmark_idx])

    return dist, dist_5


def compute_perpendicular_dist(preds, meta):
    targets = meta['origin_pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = targets.shape[1]
    norm_dists = np.zeros([N])
    norm_dists5 = np.zeros([5, N])

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')

        # curve
        k = 0
        curve2landmark68 = {
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
            11: np.arange(65, 68)
        }
        curve2landmark44 = {
            0: np.arange(0, 5),
            1: np.arange(5, 9),
            2: np.arange(9, 12),
            3: np.arange(12, 15),
            4: np.arange(15, 17),
            5: np.arange(17, 20),
            6: np.arange(20, 26),
            7: np.arange(26, 32),
            8: np.arange(32, 36),
            9: np.arange(36, 39),
            10: np.arange(39, 42),
            11: np.arange(42, 44)
        }
        landmark2curve68 = {}
        for curve_idx, curve in curve2landmark68.items():
            for landmark_idx in curve:
                landmark2curve68[landmark_idx] = curve_idx
        landmark2curve44 = {}
        for curve_idx, curve in curve2landmark44.items():
            for landmark_idx in curve:
                landmark2curve44[landmark_idx] = curve_idx

        # bilateral
        if pts_pred.shape[0] == 44:
            dist1, dist1_5 = point2curve(pts_gt, pts_pred, curve2landmark44, landmark2curve68)
            dist2, dist2_5 = point2curve(pts_pred, pts_gt, curve2landmark68, landmark2curve44)
        elif pts_pred.shape[0] == 68:
            dist1, dist1_5 = point2curve(pts_gt, pts_pred, curve2landmark68, landmark2curve68)
            dist2, dist2_5 = point2curve(pts_pred, pts_gt, curve2landmark68, landmark2curve68)

        norm_dists[i] = np.mean(np.concatenate([dist1, dist2])) / interocular
        for j in range(len(dist1_5)):
            norm_dists5[j, i] = np.mean(np.array(dist1_5[j] + dist2_5[j])) / interocular
        
    return norm_dists, norm_dists5


def decode_preds(output, center, scale, res):
    if len(output.size()) == 4:
        coords = get_preds(output)  # float type
    else:
        coords = output
    
    coords = coords.cpu()
    # pose-processing
    if len(output.size()) == 4:
        for n in range(coords.size(0)):
            for p in range(coords.size(1)):
                hm = output[n][p]
                px = int(math.floor(coords[n][p][0]))
                py = int(math.floor(coords[n][p][1]))
                if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                    diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                    coords[n][p] += diff.sign() * .25
        coords += 0.5
        
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds


class LDDMMError(nn.Module):
    def __init__(self):
        super(LDDMMError, self).__init__()
        self.sigmaW2 = np.array([18**2, 16**2, 5.5**2, 5.5**2, 3.5**2, 3.2**2,
                                 3.2**2, 3.2**2, 6**2, 4**2, 4.5**2, 1.5**2])*2.25 
        self.curve2landmark = {
            0: torch.arange(0, 9).long().cuda(),
            1: torch.arange(9, 17).long().cuda(),
            2: torch.arange(17, 22).long().cuda(),
            3: torch.arange(22, 27).long().cuda(),
            4: torch.arange(27, 31).long().cuda(),
            5: torch.arange(31, 36).long().cuda(),
            6: torch.arange(36, 42).long().cuda(),
            7: torch.arange(42, 48).long().cuda(),
            8: torch.arange(48, 55).long().cuda(),
            9: torch.arange(55, 60).long().cuda(),
            10: torch.arange(60, 65).long().cuda(),
            11: torch.arange(65, 68).long().cuda()}
        # self.curve2landmark = {
        #     0: torch.arange(0, 5).long().cuda(),
        #     1: torch.arange(5, 9).long().cuda(),
        #     2: torch.arange(9, 12).long().cuda(),
        #     3: torch.arange(12, 15).long().cuda(),
        #     4: torch.arange(15, 17).long().cuda(),
        #     5: torch.arange(17, 20).long().cuda(),
        #     6: torch.arange(20, 26).long().cuda(),
        #     7: torch.arange(26, 32).long().cuda(),
        #     8: torch.arange(32, 36).long().cuda(),
        #     9: torch.arange(36, 39).long().cuda(),
        # }

    def curve_loss(self, pred, gt, sigmaW2):
        batch_size = pred.size(0)
        n_landmark = pred.size(1)

        c_pred = (pred[:, 0:-1] + pred[:, 1:]) / 2
        tau_pred = pred[:, 1:] - pred[:, 0:-1]
        c_gt = (gt[:, 0:-1] + gt[:, 1:]) / 2
        tau_gt = gt[:, 1:] - gt[:, 0:-1]
        c = torch.cat([c_pred, c_gt], dim=1)
        tau = torch.cat([tau_pred, -tau_gt], dim=1)
        # broadcast
        c = torch.cat([c.view(batch_size, 1, -1, 2)]*(n_landmark-1)*2, dim=1)
        c_loca = c.permute(0, 2, 1, 3)
        tau = torch.cat([tau.view(batch_size, 1, -1, 2)]*(n_landmark-1)*2, dim=1)
        tau_loca = tau.permute(0, 2, 1, 3)
        # gaussian operator
        weight = torch.exp(-torch.sum((c_loca - c) ** 2, dim=-1) / sigmaW2)
        dot_tau_loca = torch.sum(tau_loca * tau, dim=-1)
        R = torch.sum((weight * dot_tau_loca).view(batch_size, -1), dim=-1)

        return R

    def forward(self, pred, gt):
        n_pt = gt.size(1)
        pred_landmarks = pred.view(pred.size(0), -1, 2)
        gt_landmarks = gt.view(pred.size(0), -1, 2)

        # landmark error
        mean_error = torch.mean(torch.linalg.norm(pred_landmarks - gt_landmarks, dim=-1), dim=-1)
        if n_pt == 68:
            eye_dist = gt_landmarks[:, 36] - gt_landmarks[:, 45]
        elif n_pt == 44:
            eye_dist = gt_landmarks[:, 20] - gt_landmarks[:, 29]
        eye_dist = torch.linalg.norm(eye_dist, dim=-1)
        landmark_error = torch.mean(mean_error / eye_dist, dim=0)
        # curve error
        curve_error = 0
        for k, v in self.curve2landmark.items():
            curve_error += self.curve_loss(pred_landmarks[:, v], gt_landmarks[:, v], self.sigmaW2[k])
        curve_error = torch.mean(curve_error / (n_pt * eye_dist), dim=0)

        return 0.8 * landmark_error + 0.2 * curve_error