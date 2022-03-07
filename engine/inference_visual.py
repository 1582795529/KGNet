# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine
import cv2
import os
from .visual import calc_metrics,torch_vis_color,save_featmap,draw_heatmap_gaussi,apply_attention,showPointSingleModal,showclassifier
from data.solver import solve_invisible_heatmap
import matplotlib.pyplot as plt
from utils.reid_metric import R1_mAP, R1_mAP_reranking
from data import get_gt_landmark_pair,save_current_SR,load_test_query,load_test_gallary,get_current_visuals
from modeling.modules.srfbn_hg_arch import merge_heatmap_4



def create_supervised_evaluator(name_list,total_id_set,feat_total,id_total,psnr_list,ssim_list,model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)


    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            hr, lr, lr_hr, landmark, gt_heatmap, pids, camid, img_path = batch
            hr = hr.to(device) if torch.cuda.device_count() >= 1 else hr
            lr = lr.to(device) if torch.cuda.device_count() >= 1 else lr
            lr_hr = lr_hr.to(device) if torch.cuda.device_count() >= 1 else lr_hr
            save_dir = '/home/fei/fei/firstwork/picture/554'
            # load_test_query(model_test_query,model)
            # load_test_gallary(model_test_gallary, model)
            name = img_path[0].split('/')[-1].split('.')[0]
            id = pids[0]
            if img_path[0].split('/')[-2] == 'image_query':
                if id in total_id_set and name not in name_list:
                    print(len(id_total), name)
                    feat, visual_feat, sr, heatmap = model.Baseline_LR(lr_hr, lr)
                    feat_total.append(feat)
                    id_total.append(id)
                    name_list.append(name)

            elif img_path[0].split('/')[-2] == 'image_test':
                if id in total_id_set and name not in name_list:
                        print("ccccc",len(id_total),name)
                        visual_feat, feat1 = model.Baseline_HR(hr)
                        # feat1 = model.Baseline_HR(hr)
                        feat_total.append(feat1)
                        id_total.append(id)
                        name_list.append(name)

                    # return feat, pids, camid


    engine = Engine(_inference)


    # for name, metric in metrics.items():
    #     metric.attach(engine, name)

    return engine,feat_total,id_total


def inference_visual(
        cfg,
        model,
        val_loader,
        num_query,
        psnr,
        ssim,
        ssim_max,
        feat_total,
        id,
        total_id_set,
        name_list
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator,feat,id = create_supervised_evaluator(name_list,total_id_set,feat_total,id,psnr,ssim,model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator,feat,id = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    # cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info(len(feat))
    logger.info(len(id))
    feat1 = feat[0]
    print(type(feat1))
    for i in range(1,len(feat)):
        feat1 = torch.cat((feat1,feat[i]),dim=0)
    showclassifier(feat1,id,r'E:\Dataset_test\firstwork\picture\classifer\motivationours.png')
    # logger.info(sum(psnr)/len(psnr))
    # logger.info(sum(ssim) / len(ssim))
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
