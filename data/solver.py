import  numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2

from PIL import Image
from torchvision import transforms
import torchvision.utils as thutil
to_image = transforms.ToPILImage()

import os

unnorm = transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225),
                                           (1/0.229, 1/0.224, 1/0.225))
lr_hr = transforms.Compose([
            transforms.Resize([128,128]),
            transforms.ToTensor()
        ])

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
            else:
                heatmaps[i,j] = np.maximum(heatmaps[i,j],0)
    heatmaps = torch.from_numpy(heatmaps).to('cuda')
    return heatmaps


def save_current_SR(hr,lr,sr):
    """
    save visual results for comparison
    """
    visuals = get_current_visual(hr,lr,sr)
    # print(visuals['HR'].shape,visuals['LR'].shape,visuals['SR'].shape)
    visuals['HR'] = visuals['HR']*255
    #visuals['LR'] = visuals['LR']*255
    visuals['SR'] = visuals['SR']*255
    visuals_list = [visuals['HR']]
    img_lr = to_image(visuals['LR'])
    #img_lr.show()
    #
    lr = lr_hr(img_lr)
    lr = lr*255
    visuals_list.extend([lr])


    visuals_list.extend([visuals['SR']])
    #print(visuals['SR'])
    visual_images = torch.stack(visuals_list)
    #print(visual_images.shape)
    #quantize(visual_images, rgb_range=1)

    visual_images = thutil.make_grid(visual_images, nrow=len(visuals_list), padding=5)
    visual_images = visual_images.byte().permute(1, 2, 0).numpy()
    #print('sssssss', visual_images.dtype)
    # visual_images = visual_images.byte().numpy()

    # cv2.imwrite(
    #     os.path.join("./outputs/img/sr", 'SR_step.png'),
    #     visual_images[:, :, ::-1])  # rgb2bgr
    return visual_images




def get_gt_landmark_pair(hr,gt_heatmap,heatmap):

    # print(hr.size())

    visuals = get_landmark_visual(hr)
    HR = visuals['HR']
    # print('aa',HR.shape)
    # print(heatmap.shape)
    gt_heatmap = gt_heatmap.cpu().detach().numpy()[0]
    heatmap = heatmap.cpu().detach().numpy()[0]
    gt_heatmap = torch.from_numpy(gt_heatmap)
    gt_heatmap = gt_heatmap.unsqueeze(0)
    heatmap = torch.from_numpy(heatmap)
    heatmap = heatmap.unsqueeze(0)

    heatmap_List = [gt_heatmap]
    heatmap_List.extend([heatmap])

    landmark = [np.squeeze(np.array(get_peak_points(h.detach().numpy()))) * 4 for h in heatmap_List]



    # print('1111',type(landmark1))
    # print('22222',HR.shape)
    # print(landmark1[0])
    fig = plot_landmark_compare(landmark, HR)
    return fig

# def get_gt_landmark_pair(cfg,hr,heatmap):
#
#     # print(hr.size())
#
#     visuals = get_landmark_visual(cfg,hr)
#     HR = visuals['HR']
#     # print('aa',HR.shape)
#     # print(heatmap.shape)
#     heatmap_List = [heatmap]
#
#
#     landmark1 = [np.squeeze(np.array(get_peak_points(h.cpu().detach().numpy()))) * 4 for h in heatmap_List]
#
#     # print('1111',type(landmark1))
#     # print('22222',HR.shape)
#     # print(landmark1[0])
#     fig = plot_landmark_compare(landmark1[0], HR)
#     return fig










def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N, 5, 32, 32)
    :return: numpy array (N, 5, 2)
    """
    #print(heatmaps.shape)
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def plot_landmark_compare(landmark, img):
    '''
    plot landmarks in faces
    :param landmark: list of numpy array (N, 2)
    :param img: list of image array (128, 128, 3) SR image
    :param img_gt: list of image array (128, 128, 3) image ground truth
    '''
    # w, h = img[0].shape[:2]
    # all_landmarks = []
    # k = 0
    # for i, l in enumerate(landmark):
    #     if k==0:
    #         biased = l + np.array([w * i, 0]).astype(float)
    #         all_landmarks.append(biased)
    #         k+=1
    # all_landmarks = np.concatenate(all_landmarks, axis=0)
    #
    # img_list = [img]
    # img_s = np.concatenate(img_list, axis=1)
    #
    # fig_withlm = plt.figure(figsize=(4 * len(img_list), 4))
    #
    # plt.imshow(img_s)
    # plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], linewidths=0.5, c='red')
    # plt.axis('off')
    # return fig_withlm
    w, h = img[0].shape[:2]
    all_landmarks = []
    for i, l in enumerate(landmark):
        biased = l + np.array([w * (i), 0]).astype(float)
        all_landmarks.append(biased)
    all_landmarks = np.concatenate(all_landmarks, axis=0)
    landmark = np.concatenate(landmark, axis=0)

    img_list = [img]
    # img_list.extend(img_list)
    # img_list.extend([img])
    img_s = np.concatenate(img_list, axis=1)

    fig_withlm = plt.figure(figsize=(4 * len(img_list), 4))

    plt.imshow(img_s)
    # plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], linewidths=0.5, c='red')
    # for i in range(20):
    #     if all_landmarks[i][0] == 0 or all_landmarks[i][1] == 0:
    #         continue
    #     plt.scatter(all_landmarks[i][0], all_landmarks[i][1], linewidths=0.5, c='red')

    # for i in range(20, 40):
    #     # if all_landmarks[i][0] == 0 or all_landmarks[i][1] == 0:
    #     #     continue
    #     plt.scatter(all_landmarks[i][0], all_landmarks[i][1], linewidths=0.5, c='blue')
    # for i in range(20):
    #     if landmark[i][0] == 0 and landmark[i][1] == 0:
    #         continue
    #     plt.scatter(landmark[i][0], landmark[i][1], linewidths=0.5, c='red')
    # for i in range(20, 40):
    #     if landmark[i][0] == 0 and landmark[i][1] == 0:
    #         continue
    #     plt.scatter(landmark[i][0], landmark[i][1], linewidths=0.5, c='blue')

    plt.axis('off')
    return fig_withlm

def get_current_visual(hr,lr,sr):
        """
        return LR image and SR list (HR) images
        """
        def _get_data(x):
            return unnorm(x[0]).data.float().cpu().numpy()

        out_dict = OrderedDict()
        out_dict['HR'] = _get_data(hr)
        out_dict['LR'] = _get_data(lr)
        out_dict['SR'] = _get_data(sr)
        # out_dict['HR'] = Tensor2np([out_dict['HR']])[0]
        # out_dict['LR'] = Tensor2np([out_dict['LR']])[0]
        # out_dict['SR'] = Tensor2np([out_dict['SR']])[0]
        return out_dict


def get_current_visuals(hr, lr, sr):
    """
    return LR image and SR list (HR) images
    """

    def _get_data(x):
        return unnorm(x[0]).data.float().cpu()

    out_dict = OrderedDict()

    out_dict['LR'] = _get_data(lr)
    out_dict['SR'] = _get_data(sr)
    out_dict['HR'] = _get_data(hr)


    out_dict['LR'] = Tensor2np(
            [out_dict['LR']])
    out_dict['SR'] = Tensor2np(
            [out_dict['SR']])

    out_dict['HR'] = Tensor2np([out_dict['HR']])

    return out_dict


# def get_landmark_visual(hr):
#     """
#     return LR image and SR list (HR) images
#     """
#
#     def _get_data(x):
#         return unnorm(x[0]).data.float().cpu()
#
#     out_dict = OrderedDict()
#     out_dict['HR'] = _get_data(hr)
#
#     out_dict['HR'] = Tensor2np([out_dict['HR']])[0]
#     # out_dict['LR'] = Tensor2np([out_dict['LR']])[0]
#     # out_dict['SR'] = Tensor2np([out_dict['SR']])[0]
#     return out_dict

def get_landmark_visual(hr):
    """
    return LR image and SR list (HR) images
    """
    def _get_data(x):
        return unnorm(x[0]).data.float().cpu()

    out_dict = OrderedDict()
    out_dict['HR'] = _get_data(hr)

    out_dict['HR'] = Tensor2np([out_dict['HR']])[0]
    # out_dict['LR'] = Tensor2np([out_dict['LR']])[0]
    # out_dict['SR'] = Tensor2np([out_dict['SR']])[0]
    return out_dict


def Tensor2np(tensor_list):

    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor,rgb_range=1) for tensor in tensor_list]


def quantize(img, rgb_range):
    pixel_range = 255. / rgb_range
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, 255).round()

def load_test_query(model_lr,model):

    for k in model.state_dict():
        if 'Baseline_LR' in k:
            v = k[12:]
            model_lr.state_dict()[v].copy_(model.state_dict()[k])
        else:
            continue

def load_test_gallary(model_hr,model):

    for k in model.state_dict():
        if 'Baseline_HR' in k:
            v = k[12:]
            model_hr.state_dict()[v].copy_(model.state_dict()[k])
        else:
            continue

def load(cfg,model):
    hg_choice = cfg.MODEL.HG_PRETRAIN_CHOICE
    hg_path = cfg.MODEL.HG_PATH
    para_dict = torch.load(
        '/home/fei/firstwork/pth/landmark.pth')

    new_state_dict = OrderedDict()
    for k,v in para_dict.named_parameters():
        if 'HG' in k:
            k = k[15:]
            new_state_dict[k] = v
    if hg_choice == 'hourglass':
        model.Baseline_LR.HG.load_state_dict(new_state_dict)
        for name, para in model.Baseline_LR.HG.named_parameters():
            para.requires_grad = False
        print('Loading pretrained keypoint model......')

def load_model(cfg,model):

    para_dict = torch.load(
        './pth/kgnet.pth')
    # para_dict_sr_heatmap = torch.load(
    #     './pth/sr_landmark.pth'
    # )

    new_state_dict = OrderedDict()
    for k,v in para_dict.named_parameters(): # there dont include HG_SR
        # if 'fuse' in k:
        #     continue
        model.state_dict()[k].copy_(v)


        # sr_state_dict = torch.load(hg_path)

        # new_state_dict = OrderedDict()
        # for k, v in sr_state_dict.items():
        #
        #     new_state_dict[name] = v
        #     print('cdsssssa',name)
        #     if name in 'HG.HG.compress_in.bias':
        #         print(v.requires_grad)
        #         print('5555555', v)
    for name, para in model.named_parameters():
        if 'HG' in name:
            para.requires_grad = False
    print('Loading pretrained  model......')




