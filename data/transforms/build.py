# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
import torch
import numpy
from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform_hr = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        transform_lr = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.Resize(cfg.INPUT.LR_TRAIN),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.LR_TRAIN),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        transform_lr_hr = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            #T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.LR_TRAIN),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform_hr = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform
        ])
        transform_lr = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.Resize(cfg.INPUT.LR_TRAIN),
            T.ToTensor(),
            normalize_transform
        ])
        transform_lr_hr = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform
        ])

    return transform_hr,transform_lr,transform_lr_hr

# problem****
def build_unnorm(cfg,img):
    img = img.numpy()

    mean = numpy.array(cfg.INPUT.PIXEL_MEAN)
    std = numpy.array(cfg.INPUT.PIXEL_STD)
    img = img*std+mean
    img = torch.from_numpy(img)

    return img