# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms,build_unnorm
from torchvision import transforms as tr

to_image = tr.ToPILImage()


def make_data_loader(cfg):
    train_hr_transforms,train_lr_transforms,train_lr_hr_transforms = build_transforms(cfg, is_train=True)
    val_hr_transforms,val_lr_transforms,val_lr_hr_transforms = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_hr_transforms,train_lr_transforms,train_lr_hr_transforms)
    # img_hr,img_lr, landmark,gt_heatmap,pid, camid, img_path= train_set[10]
    # print(img_hr.size())
    # print(img_lr.size())
    # img1_hr = to_image(img_hr)
    # img1_lr = to_image(img_lr)
    # img1_hr.show()
    # img1_lr.show()
    # img_hr = to_image(build_unnorm(cfg,img_hr))
    # img_lr = to_image(build_unnorm(cfg,img_lr))
    # img_hr.show()
    # img_lr.show()
    # print(landmark)
    # print(train_set[10])

    if cfg.DATALOADER.SAMPLER == 'softmax':

        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:

        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    # val_set = ImageDataset(dataset.query + dataset.gallery, val_hr_transforms,val_lr_transforms)
    

    val_set = ImageDataset(dataset.query_gallary, val_hr_transforms, val_lr_transforms,val_lr_hr_transforms)


    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes
