import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .srfbn_hg_arch import FeedbackBlockCustom, FeedbackBlockHeatmapAttention, merge_heatmap_4,merge_heatmap_3
from .architecture import FeedbackHourGlass
from .blocks import ResBlock,ConvBlock

def solve_invisible_heatmap(heatmap):
    heatmaps = heatmap.cpu().detach().numpy()
    N, C, H, W = heatmaps.shape

    for i in range(N):
        for j in range(C):
            yy, xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            if y == 0 and x == 0:
                heat = torch.from_numpy(heatmaps[i,j])
                heat = torch.zeros_like(heat).numpy()
                heatmaps[i,j] = heat

            # else:
            #     heatmaps[i,j] = np.maximum(heatmaps[i,j],0)
    heatmaps = torch.from_numpy(heatmaps).to('cuda')
    return heatmaps





def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Fuse(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.detach_attention = cfg.INPUT.DETACH_ATTENTION
        self.in_dim = 276
        self.out_dim = 64
        self.num_heatmap = 3
        self.num_feature = 3
        self.num_block = 7
        self.res_block_channel = self.num_feature * self.num_heatmap
        self.num_mid_channel = self.num_heatmap * self.num_feature
        # self.fuse_out = nn.Conv2d(self.in_dim, self.out_dim, 1)
        self.conv_in = ConvBlock(self.num_feature, self.res_block_channel, 1, norm_type=None, act_type='lrelu')
        self.resnet = nn.Sequential(*[
            ResBlock(self.res_block_channel,
                     self.res_block_channel,
                     self.num_mid_channel,
                     3,
                     norm_type=None,
                     act_type='lrelu',
                     groups=self.num_heatmap) for _ in range(self.num_block)
        ])
    def forward(self, sr,heatmap):
          batch_size = heatmap.size(0)
          w, h = sr.shape[-2:]

          heatmap = solve_invisible_heatmap(heatmap)
          heatmap = F.interpolate(heatmap, (128, 128), mode='bilinear', align_corners=False)
          heatmap = merge_heatmap_3(heatmap,self.detach_attention)
          attention = nn.functional.softmax(heatmap, dim=1)  # 8*4*32*32
          # attention = attention.sum(1)

          feature = self.conv_in(sr)
          feature = self.resnet(feature)  # B * (num_heatmap*feat_channel_in) * h * w
          sr_feat = feature

          feature = feature.view(batch_size, self.num_heatmap, -1, w, h) * attention.unsqueeze(2)
          feature = feature.sum(1)

          return feature
          # print('111111',attention.size())
          # x = torch.cat((x,heat),dim=1)
          # x = self.fuse_out(x)
          # return x
