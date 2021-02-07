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
from ..utils.lddmm_params import get_curve2landmark, get_sigmaV2


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
        elif L == 68 or L == 72:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
            # rmse[i] = np.sum(np.linalg.norm(pts_pred[0:17:2] - pts_gt[0:17:2], axis=1)) / (interocular * 9)
        elif L == 98 or L == 96:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 46 or L == 50:  # 300w weak supverised
            # interocular
            interocular = np.linalg.norm(pts_gt[22, ] - pts_gt[31, ])
            # rmse[i] = np.sum(np.linalg.norm(pts_pred[0:9] - pts_gt[0:9], axis=1)) / (interocular * 9)
        elif L == 54:  # wflw weak supverised
            interocular = np.linalg.norm(pts_gt[34, ] - pts_gt[40, ])
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

    curve2landmark_pred = get_curve2landmark(meta['dataset_name'][0], preds.shape[1])
    curve2landmark_gt = get_curve2landmark(meta['dataset_name'][0], L)
    idx_5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10, 11]]

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68 or L == 72:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98 or L == 96:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 46 or L == 50:  # 300w weak supverised
            # interocular
            interocular = np.linalg.norm(pts_gt[22, ] - pts_gt[31, ])
        elif L == 54:  # wflw weak supverised
            interocular = np.linalg.norm(pts_gt[34, ] - pts_gt[40, ])
        else:
            raise ValueError('Number of landmarks is wrong')

        for j, curve_idxs in enumerate(idx_5):
            gt_landmark_idxs = []
            pred_landmark_idxs = []
            for curve_idx in curve_idxs:
                gt_landmark_idxs.extend(list(curve2landmark_gt[curve_idx].cpu().numpy()))
                pred_landmark_idxs.extend(list(curve2landmark_pred[curve_idx].cpu().numpy()))

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
        curve_pts = pts_set2[curve_pts_idx.cpu().numpy()]
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
        elif L == 68 or L == 72:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98 or L == 96:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 46 or L == 50:  # 300w weak supverised
            # interocular
            interocular = np.linalg.norm(pts_gt[22, ] - pts_gt[31, ])
        elif L == 54:  # wflw weak supverised
            interocular = np.linalg.norm(pts_gt[34, ] - pts_gt[40, ])
        else:
            raise ValueError('Number of landmarks is wrong')

        # curve
        k = 0
        curve2landmark_pred = get_curve2landmark(meta['dataset_name'][0], preds.shape[1])
        curve2landmark_gt = get_curve2landmark(meta['dataset_name'][0], L)
        landmark2curve_pred = {}
        for curve_idx, curve in curve2landmark_pred.items():
            for landmark_idx in curve:
                landmark2curve_pred[int(landmark_idx.cpu().numpy())] = curve_idx
        landmark2curve_gt = {}
        for curve_idx, curve in curve2landmark_gt.items():
            for landmark_idx in curve:
                landmark2curve_gt[int(landmark_idx.cpu().numpy())] = curve_idx

        # bilateral
        dist1, dist1_5 = point2curve(pts_gt, pts_pred, curve2landmark_pred, landmark2curve_gt)
        dist2, dist2_5 = point2curve(pts_pred, pts_gt, curve2landmark_gt, landmark2curve_pred)

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


def decode_duplicate(preds, config):
    if config.MODEL.NUM_JOINTS == 72:
        new_preds = torch.zeros([preds.size(0), 68, 2])
        new_preds[:, 0:48] = preds[:, 0:48]
        # new_preds[:, 48] = (preds[:, 48] + preds[:, 61]) / 2
        new_preds[:, 48] = preds[:, 61]
        new_preds[:, 49:54] = preds[:, 49:54]
        # new_preds[:, 54] = (preds[:, 54] + preds[:, 55]) / 2 
        new_preds[:, 54] = preds[:, 55]
        new_preds[:, 55:60] = preds[:, 56:61]
        # new_preds[:, 60] = (preds[:, 62] + preds[:, 71]) / 2
        new_preds[:, 60] = preds[:, 71]
        new_preds[:, 61:64] = preds[:, 63:66]
        # new_preds[:, 64] = (preds[:, 66] + preds[:, 67]) / 2
        new_preds[:, 64] = preds[:, 67]
        new_preds[:, 65:68] = preds[:, 68:71]
        preds = new_preds.clone()
    elif config.MODEL.NUM_JOINTS == 50:
        new_preds = torch.zeros([preds.size(0), 46, 2])
        new_preds[:, 0:34] = preds[:, 0:34]
        new_preds[:, 34] = (preds[:, 34] + preds[:, 43]) / 2
        new_preds[:, 35:38] = preds[:, 35:38]
        new_preds[:, 38] = (preds[:, 38] + preds[:, 39]) / 2 
        new_preds[:, 39:42] = preds[:, 40:43]
        new_preds[:, 42] = (preds[:, 44] + preds[:, 49]) / 2
        new_preds[:, 43] = preds[:, 45]
        new_preds[:, 44] = (preds[:, 46] + preds[:, 47]) / 2
        new_preds[:, 45] = preds[:, 48]
        preds = new_preds.clone()

    return preds


class LDDMMError(nn.Module):
    def __init__(self, config, curve=True):
        super(LDDMMError, self).__init__()
        self.sigmaW2 = get_sigmaV2(config.DATASET.DATASET, config.MODEL.NUM_JOINTS) / 4
        self.curve2landmark = get_curve2landmark(config.DATASET.DATASET, config.MODEL.NUM_JOINTS)
        self.curve = curve

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
        if n_pt == 68 or n_pt ==72:
            eye_dist = gt_landmarks[:, 36] - gt_landmarks[:, 45]
        elif n_pt == 46 or n_pt == 50:
            eye_dist = gt_landmarks[:, 22] - gt_landmarks[:, 31]
        elif n_pt == 98 or n_pt == 96:
            eye_dist = gt_landmarks[:, 60] - gt_landmarks[:, 72]
        elif n_pt == 54:
            eye_dist = gt_landmarks[:, 34] - gt_landmarks[:, 40]


        eye_dist = torch.linalg.norm(eye_dist, dim=-1)
        landmark_error = torch.mean(mean_error / eye_dist, dim=0)

        if not self.curve:
            return landmark_error

        # curve error
        curve_error = 0
        for k, v in self.curve2landmark.items():
            curve_error += self.curve_loss(pred_landmarks[:, v], gt_landmarks[:, v], self.sigmaW2[k])
        curve_error = torch.mean(curve_error / (n_pt * eye_dist), dim=0)

        return 0.8 * landmark_error + 0.2 * curve_error