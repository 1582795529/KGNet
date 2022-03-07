# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn


from .modules.baseline_hr import Baseline_HR
from .modules.baseline_lr import Baseline_LR
from torchvision import transforms
to_img = transforms.ToPILImage()

from .modules.dic_arch import DIC

from collections import OrderedDict


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, cfg):
        super(Baseline, self).__init__()

        neck = cfg.MODEL.NECK
        neck_feat = cfg.TEST.NECK_FEAT



        self.num_steps = cfg.INPUT.NUM_STEPS
        num_features = cfg.INPUT.NUM_FEATURE
        self.upscale_factor = cfg.INPUT.SCALE
        self.detach_attention = cfg.INPUT.DETACH_ATTENTION


        self.num_classes = num_classes

        self.Baseline_HR = Baseline_HR(cfg)
        self.Baseline_LR = Baseline_LR(cfg)



        if self.detach_attention:
            print('Detach attention!')
        else:
            print('Not detach attention!')



        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)


    def forward(self, x,y,z):   # x : HR    y : LR    z:LR->HR

        if self.training:
            global_feat_hr,feat_hr = self.Baseline_HR(x)
            global_feat_lr, feat_lr ,sr,heatmap= self.Baseline_LR(y,z)
            score_hr = self.classifier(feat_hr)
            score_lr = self.classifier(feat_lr)
            return score_hr, global_feat_hr,score_lr,global_feat_lr,heatmap,sr # global feature for triplet loss
        else:
            feat_hr = self.Baseline_HR(x)
            feat_lr,sr,heatmap = self.Baseline_LR(z,y)
            return feat_lr,feat_hr,heatmap,sr

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def load_model_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict.state_dict():
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict.state_dict()[i])


