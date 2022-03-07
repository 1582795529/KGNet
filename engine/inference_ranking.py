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
from .visual import calc_metrics,torch_vis_color,save_featmap,draw_heatmap_gaussi,apply_attention,showPointSingleModal,rank_visual
from data.solver import solve_invisible_heatmap
import matplotlib.pyplot as plt
from utils.reid_metric import R1_mAP, R1_mAP_reranking
from data import get_gt_landmark_pair,save_current_SR,load_test_query,load_test_gallary,get_current_visuals
from modeling.modules.srfbn_hg_arch import merge_heatmap_4



def create_supervised_evaluator(model, qf, ql, qc, qi, gf, gl, gc, gi, metrics,
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
            save_dir = r'E:\Dataset_test\firstwork\picture\feature_map\KG120'
            name = img_path[0].split('/')[-1].split('.')[0]
            id = pids[0]

            if img_path[0].split('/')[-2] == 'image_query':
                feat, visual_feat, sr, heatmap = model.Baseline_LR(lr_hr, lr)
                qf.append(feat)
                ql.append(pids)
                qc.append(camid)
                qi.append(img_path)

                return feat, pids, camid

            elif img_path[0].split('/')[-2] == 'image_test':
                visual_feat,feat = model.Baseline_HR(hr)
                gf.append(feat)
                gl.append(pids)
                gc.append(camid)
                gi.append(img_path)

                return feat, pids, camid


    engine = Engine(_inference)


    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine,qf, ql, qc, qi, gf, gl, gc, gi


def inference_ranking(
        cfg,
        model,
        val_loader,
        num_query,
        qf,
        ql,
        qc,
        qi,
        gf,
        gl,
        gc,
        gi
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator,qf, ql, qc, qi, gf, gl, gc, gi = create_supervised_evaluator(model, qf, ql, qc, qi, gf, gl, gc, gi, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))



    evaluator.run(val_loader)
    qf1 = qf[0]
    for i in range(1, len(qf)):
        qf1 = torch.cat((qf1, qf[i]), dim=0)
    gf1 = gf[0]
    g_i = []
    g_i.append(gi[0])
    for i in range(1, len(gf)):
        if gi[i] not in g_i:
            gf1 = torch.cat((gf1, gf[i]), dim=0)
            g_i.append(gi[i])
    for i in range(len(qf)):
        rank_visual(i,qf1,ql,qc,qi,gf1,gl,gc,gi)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')

    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
