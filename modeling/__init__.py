# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .modules.baseline_lr import Baseline_LR
from .modules.baseline_hr import Baseline_HR


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = Baseline(num_classes, cfg)
    return model

def build_model_testquery(cfg):
    model_test = Baseline_LR( cfg)
    return model_test

def build_model_testgallary(cfg):
    model_test = Baseline_HR(cfg)
    return model_test
