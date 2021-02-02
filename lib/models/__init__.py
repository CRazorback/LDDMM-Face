
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng (tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hrnet import HighResolutionNet
from .lddmm_hrnet import LDDMMHighResolutionNet
from .refine_hrnet import RefineHighResolutionNet
from .fcfan_hrnet import FCFAN
from .fcfan_share import RFCFAN


def get_face_alignment_net(config, **kwargs):
    if config.MODEL['NAME'] == 'hrnet':
        model = HighResolutionNet(config, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'lddmm_hrnet':
        model = LDDMMHighResolutionNet(config, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'refine_hrnet':
        model = RefineHighResolutionNet(config, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.hrnet.init_weights(pretrained=pretrained)
        model.lddmm_hrnet.init_weights(pretrained=pretrained)
    elif config.MODEL['NAME'] == 'fcfan':
        model = FCFAN(config, **kwargs)
        pretrained1 = config.MODEL.PRETRAINED1 if config.MODEL.INIT_WEIGHTS else ''
        pretrained2 = config.MODEL.PRETRAINED2 if config.MODEL.INIT_WEIGHTS else ''
        model.stage1.init_weights(pretrained=pretrained1)
        model.stage2.init_weights(pretrained=pretrained2)
    elif config.MODEL['NAME'] == 'rfcfan':
        model = RFCFAN(config, **kwargs)
        pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
        model.stage1.init_weights(pretrained=pretrained)
        model.stage2.init_weights(pretrained=pretrained)
    else:
        raise NotImplementedError('{} is not available'.format(config.model['NAME']))

    return model

__all__ = ['HighResolutionNet', 'LDDMMHighResolutionNet', 'RefineHighResolutionNet'
           'FCFAN', 'RFCFAN', 'get_face_alignment_net']
