import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock
from .srfbn_hg_arch import FeedbackBlockCustom, FeedbackBlockHeatmapAttention, merge_heatmap_4
from torchvision import transforms

to_img = transforms.ToPILImage()



class DIC(nn.Module):
    def __init__(self, cfg,device):
        super().__init__()
        in_channels = cfg.INPUT.IN_CHANNELS
        out_channels = cfg.INPUT.OUT_CHANNELS
        num_groups = cfg.INPUT.NUM_GROUPS
        num_fusion_block = cfg.INPUT.NUM_FUSION_BLOCK
        act_type = 'prelu'
        norm_type = None

        self.num_steps = cfg.INPUT.NUM_STEPS
        num_features = cfg.INPUT.NUM_FEATURE
        self.upscale_factor = cfg.INPUT.SCALE
        self.detach_attention = cfg.INPUT.DETACH_ATTENTION


        if self.upscale_factor == 8:
            # with PixelShuffle at start, need to upscale 4x only
            stride = 4
            padding = 2
            kernel_size = 8
        else:
            raise NotImplementedError("Upscale factor %d not implemented!" % self.upscale_factor)

        # LR feature extraction block
        self.conv_in = ConvBlock(
            in_channels,
            4*num_features,
            kernel_size=3,
            act_type=act_type,
            norm_type=norm_type)
        self.feat_in = nn.PixelShuffle(2)

        # basic block
        self.first_block = FeedbackBlockCustom(num_features, num_groups, self.upscale_factor,
                                   act_type, norm_type, num_features)
        self.block = FeedbackBlockHeatmapAttention(num_features, num_groups, self.upscale_factor, act_type, norm_type, 4, num_fusion_block, device=device)
        self.block.should_reset = False

        # reconstruction block
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = DeconvBlock(
            num_features,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_type='prelu',
            norm_type=norm_type)
        self.conv_out = ConvBlock(
            num_features,
            out_channels,
            kernel_size=3,
            act_type=None,
            norm_type=norm_type)



    def forward(self, heatmap,x):  # heatmap : 32*32*3    y : 16*16*3



        inter_res = nn.functional.interpolate(
            x,
            scale_factor=self.upscale_factor,
            mode='bilinear',
            align_corners=False)

        x = self.conv_in(x)
        self.self_feat_in = self.feat_in(x)
        x = self.self_feat_in

        FB_out = self.block(x, merge_heatmap_4(heatmap, self.detach_attention))
        sr = torch.add(inter_res, self.conv_out(self.out(FB_out)))

        return sr  # return output of every timesteps
