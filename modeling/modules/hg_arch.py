import torch
import torch.nn as nn

from .architecture import FeedbackHourGlass

from torchvision import transforms
to_img = transforms.ToPILImage()

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

class HG(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        hg_num_feature = cfg.INPUT.HG_NUM
        num_vehicle_keypoint = cfg.INPUT.KEYPOINT_NUM

        self.HG = FeedbackHourGlass(hg_num_feature, num_vehicle_keypoint)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.bottleneck = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck.bias.requires_grad_(False)  # no shift
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #
        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)


    def forward(self, x):
          heatmap = self.HG(x)
          # global_feat = self.gap(feature)  # global_feat B*512*1*1
          # global_feat = global_feat.view(global_feat.shape[0], -1)  # global_feat B*512
          # feat = self.bottleneck(global_feat)
          # pose = self.classifier(feat)

          return heatmap  #,pose   # return output of every timesteps
