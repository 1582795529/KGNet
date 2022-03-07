# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage import io
import torch
from  torchvision import  transforms



to_img = transforms.ToPILImage()


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def generate_gt(size, landmark_list, sigma):
    '''
    return N * H * W
    '''
    heatmap_list = [
        _generate_one_heatmap(size, l, sigma) for l in landmark_list
    ]
    return np.stack(heatmap_list, axis=0)

def _generate_one_heatmap(size, landmark, sigma):
    w, h = size
    x_range = np.arange(start=0, stop=w, dtype=int)
    y_range = np.arange(start=0, stop=h, dtype=int)
    xx, yy = np.meshgrid(x_range, y_range)
    # print('333',xx)

    d2 = (xx - landmark[0])**2 + (yy - landmark[1])**2
    np.set_printoptions(threshold=100000)
    # print('***************')
    # print(d2)
    # print('--------------------')
    # print(landmark[0], landmark[1])
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    # print('d2',d2[0][0],'exposent',exponent[0][0],'heatmap',heatmap[0][0])
    # print('\n')
    # i = int(landmark[0])
    # j = int(landmark[1])
    # print('landmark',i,j,'d2', d2[j][i], 'exposent', exponent[j][i], 'heatmap', heatmap[j][i])
    # print('\n')
    return heatmap




class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform_hr = None,transform_lr = None,transform_lr_hr = None):
        self.dataset = dataset
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.transform_lr_hr = transform_lr_hr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        img_path,landmark_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        image = io.imread(img_path)
        image = image.astype(np.float)


        landmark = np.zeros([20, 2])
        for i in range(0, 20):
            landmark[i] = [int(b) for b in landmark_path[index].split(' ')[2 * i + 1: 2 * i + 3]]

        pose = int(landmark_path[index].split(' ')[-1])

        H, W = image.shape[0], image.shape[1]

        # Resize annotations to fit the modifed image
        landmark[:, 0] = landmark[:, 0] * (128 / W)
        landmark[:, 1] = landmark[:, 1] * (128 / H)

        keypoint = [(x[0] / 4, x[1] / 4) for x in landmark]


        # gt_heatmaps = np.zeros([20, 32, 32])
        #
        # for i, pt in enumerate(landmark):
        #
        #     if (32>=pt[0] > 0) and (32>=pt[1] > 0):
        #         gt_heatmaps[i][int(pt[1])][int(pt[0])] = 1

        gt_heatmaps = generate_gt(
            (32,32), keypoint, 1)


        # print('sss',type(gt_heatmap),gt_heatmap.size())

        if self.transform_hr is not None:
            img_hr = self.transform_hr(img)
        if self.transform_lr is not None:
            img_lr = self.transform_lr(img)
        if self.transform_lr_hr is not None:
            img_lr_hr = self.transform_lr_hr(img)

        gt_heatmap = torch.from_numpy(np.ascontiguousarray(gt_heatmaps))
        return img_hr, img_lr,img_lr_hr, landmark, gt_heatmap, pid, camid, img_path
