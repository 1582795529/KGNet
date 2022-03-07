# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    img_hr,img_lr,img_lr_hr , landmark,gt_heatmap,pids, camid, img_path = zip(*batch)

    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(img_hr, dim=0),torch.stack(img_lr, dim=0),torch.stack(img_lr_hr, dim=0),landmark,torch.stack(gt_heatmap, dim=0), pids,camid


def val_collate_fn(batch):
    img_hr,img_lr,img_lr_hr, landmark,gt_heatmap,pids, camid, img_path= zip(*batch)
    return torch.stack(img_hr, dim=0),torch.stack(img_lr, dim=0),torch.stack(img_lr_hr, dim=0),landmark,torch.stack(gt_heatmap, dim=0), pids,camid,img_path
