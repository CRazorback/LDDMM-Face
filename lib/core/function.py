# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import numpy as np

from .evaluation import compute_curve_dist, compute_perpendicular_dist, decode_preds, decode_duplicate, compute_nme
from ..utils.transforms import transform_preds

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)

        # auxilary 
        if isinstance(output, list):
            output, output_aux = output[1], output[0]
            loss = 0.7 * critertion(output, target) + 0.3 * critertion(output_aux, target)
        else:
            loss = critertion(output, target)

        # NME
        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)

        nme_batch = compute_nme(preds, meta, config)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            inp = inp.cuda(non_blocking=True)  
            target = target.cuda(non_blocking=True)
            data_time.update(time.time() - end)
            output = model(inp)

            if isinstance(output, list):
                output = output[1]
                
            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)
            preds = decode_duplicate(preds, config)
                
            # NME
            nme_temp = compute_nme(preds, meta, config)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    curve_dist_batch_sum = 0
    curve_dist5_batch_sum = [0, 0, 0, 0, 0]
    pcurve_dist_batch_sum = 0
    pcurve_dist5_batch_sum = [0, 0, 0, 0, 0]
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            inp = inp.cuda()
            data_time.update(time.time() - end)
            output = model(inp)
            if isinstance(output, list):
                output = output[1]
                
            score_map = output.data.cpu()

            preds = decode_preds(score_map, meta['center'], meta['scale'], config.MODEL.HEATMAP_SIZE)
            preds = decode_duplicate(preds, config)

            # NME
            nme_temp = compute_nme(preds, meta, config)
            curve_dist_temp, curve_dist5_temp = compute_curve_dist(preds, meta)
            pcurve_dist_temp, pcurve_dist5_temp = compute_perpendicular_dist(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            curve_dist_batch_sum += np.sum(curve_dist_temp)
            pcurve_dist_batch_sum += np.sum(pcurve_dist_temp)
            for j in range(len(curve_dist5_batch_sum)):
                curve_dist5_batch_sum[j] += np.sum(curve_dist5_temp[j])
                pcurve_dist5_batch_sum[j] += np.sum(pcurve_dist5_temp[j])
            nme_count = nme_count + preds.size(0)
            # for n in range(score_map.size(0)):
            #     predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    curve_dist = curve_dist_batch_sum / nme_count
    curve_dist5 = np.array(curve_dist5_batch_sum) / nme_count
    pcurve_dist = pcurve_dist_batch_sum / nme_count
    pcurve_dist5 = np.array(pcurve_dist5_batch_sum) / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    msg2 = 'Curve error:{:.5f} [Edge]:{:.5f} [Eyebrow]:{:.5f} [Nose]:{:.5f} [Eye]:{:.5f} [Mouth]:{:.5f}' \
            .format(curve_dist, curve_dist5[0], curve_dist5[1], curve_dist5[2], curve_dist5[3], curve_dist5[4])
    msg3 = 'P-Curve error:{:.5f} [Edge]:{:.5f} [Eyebrow]:{:.5f} [Nose]:{:.5f} [Eye]:{:.5f} [Mouth]:{:.5f}' \
            .format(pcurve_dist, pcurve_dist5[0], pcurve_dist5[1], pcurve_dist5[2], pcurve_dist5[3], pcurve_dist5[4])
    logger.info(msg)
    logger.info(msg2)
    logger.info(msg3)

    return nme, predictions



