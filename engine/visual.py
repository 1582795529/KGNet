import os
import  torch
import math
from datetime import datetime
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import  numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.manifold import TSNE
from torchvision import transforms


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                           [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

####################
# metric
####################
def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2

    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')




# draw feature map
def torch_vis_color(name, pic, feature_tensor, col, raw, save_path, colormode=2, margining=1):
    '''
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11
    :param feature_tensor: torch.Tensor [1,c,w,h]
    :param col: col num
    :param raw: raw num
    :param save_path: save path
    :param colormode: cv2.COLORMAP
    :return:None
    '''
    # print(feature_tensor.shape)
    # pdb.set_trace()
    # show_k = col * raw  # total num
    # #print('111',feature_tensor.shape)
    # f = feature_tensor[0, :show_k, :, :]  # n,c,h,w
    # #print('2',f.shape)
    # size = f[0, :, :].shape  # h*w
    # #print('3333',size)
    # f = f.data.cpu().numpy()
    # fmin = np.min(f)
    # fmax = np.max(f)
    # #print(fmax, fmin)
    # for i in range(raw):
    #     f = (f - fmin) / (fmax - fmin + 0.0001)
    #     tem = f[i * col, :, :] * 255 / (np.max(f[i * col, :, :] + 1e-14))
    #     # print("tem",tem.shape)
    #     tem = cv2.applyColorMap(np.array(tem, dtype=np.uint8), colormode)
    #     for j in range(col):
    #         if not j == 0:
    #             tem = np.concatenate((tem, np.ones((size[0], margining, 3), dtype=np.uint8) * 255), 1)
    #             tem2 = cv2.applyColorMap(
    #                 np.array(f[i * col + j, :, :] * 255 / (np.max(f[i * col + j, :, :]) + 1e-14), dtype=np.uint8),
    #                 colormode)
    #             tem = np.concatenate((tem, tem2), 1)
    #     if i == 0:
    #         final = tem
    #     else:
    #         final = np.concatenate(
    #             (final, np.ones((margining, size[1] * col + (col - 1) * margining, 3), dtype=np.uint8) * 255), 0)
    #         final = np.concatenate((final, tem), 0)
    # print(final.shape)
    # cv2.imwrite(save_path+name+'.jpg',final)

    # cv2.imwrite(save_path+name+str(col)+'*'+str(raw)+'.png',final)
    # feature mean
    feature_mean = feature_tensor.mean(dim=1, keepdim=True)  # n,c,h,w
    feature_mean = feature_mean * 255 / torch.max(feature_mean + 1e-14)

    feature_mean = F.interpolate(feature_mean, size=(256, 256), mode='bilinear', align_corners=False)
    feature_mean = feature_mean.squeeze().data.cpu().numpy()

    feature_mean = cv2.applyColorMap(np.array(feature_mean, dtype=np.uint8), colormode)


    un_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    pic = un_norm(pic.data.float().cpu())
    pic = pic * 255 / torch.max(pic + 1e-14)
    pic = F.interpolate(pic, size=(256, 256), mode='bilinear', align_corners=False)
    pic = pic.squeeze().data.cpu().numpy().transpose(1,2,0)


    draw_pic = 0.8*pic + 0.4*feature_mean
    # draw_pic = draw_pic / np.max(draw_pic)
    feature_name = 'KG120_'+name+'.jpg'
    print(os.path.join(save_path, feature_name))
    cv2.imwrite(os.path.join(save_path, feature_name), draw_pic)



def save_featmap(feat, name, output_dir, colormode=2):
    # pdb.set_trace()
    feat = feat.squeeze()
    if not os.path.exists(output_dir):
        p = os.path.abspath(output_dir)
        os.mkdir(p)
        print("dir dose not exist, make it:" + p)

    shape = feat.shape
    # if len(shape) != 3:
    #     raise Exception("input feat should be a 3-dim tensor")

    C, H, W = shape
    target_H, target_W = H, W
    flag_resize = False
    if H < 32 or W < 32:
        flag_resize = True

    feat = feat.cuda().data.cpu().numpy()
    fmin = np.min(feat)
    fmax = np.max(feat)
    print(fmax, fmin)
    for i in range(C):
        # pdb.set_trace()
        map_name = name + '_c{}'.format(i)
        featmap = feat[i, :, :]
        featmap = (featmap - fmin) / (fmax - fmin + 0.0001)
        featmap = (featmap * 255).astype(np.uint8)
        featmap = cv2.applyColorMap(np.array(featmap, dtype=np.uint8), colormode)
        if flag_resize:
            featmap = cv2.resize(featmap, (W * 5, H * 5), interpolation=cv2.INTER_LINEAR)
            map_name += '_upsamp'
        map_name += '.jpg'

        cv2.imwrite(os.path.join(output_dir, map_name), featmap)


def draw_heatmap_gaussi(heatmap,name):
    #heatmap = F.interpolate(heatmap, size=(128, 128), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze(0)
    heatmaps = heatmap.cpu().numpy()
    C, H, W = heatmap.shape
    flag = 0
    for j in range(C):
        yy, xx = np.where(heatmaps[j] == heatmaps[j].max())
        y = yy[0]
        x = xx[0]
        if y == 0 and x == 0:
            if flag == 0:
                flag = 1
                continue
            heat = torch.from_numpy(heatmaps[j])
            heat = torch.zeros_like(heat).numpy()
            heatmaps[j] = heat
        # else:
        #     heatmaps[j] = np.maximum(heatmaps[i, j], 0)

    # print('11111',heatmaps.shape)
    heat = heatmaps.sum(0)
    # print('2222',heat.shape)
    # heat = heatmaps[4]
    C, H, W = heatmaps.shape

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    np.set_printoptions(threshold=100000)
    # Make data.
    # w, h = (H,W)
    # X = np.arange(start=0, stop=w, dtype=int)
    # Y = np.arange(start=0, stop=h, dtype=int)
    # X, Y = np.meshgrid(X, Y)
    X = np.arange(0, 32, 1)
    Y = np.arange(0, 32, 1)
    X, Y = np.meshgrid(X, Y)
    Z =heat
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
    # surf = ax.plot_trisurf(X, Y, Z, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.savefig(
        os.path.join('./outputs/img', 'GT_Landmark_'+name+'.png'))
    plt.close(fig)
    # plt.show()



def apply_attention(heatmap,img,sr_feat,dir,name):
    # heatmap_ori = heatmap
    # fig = get_gt_landmark_pair(img, heatmap_ori, heatmap)
    # fig.savefig(
    #     os.path.join('/home/zhang/work/pic/pic_4attention/landmark.png') )
    # plt.close(fig)
    print(sr_feat.shape)
    sr_feat = sr_feat.cpu().numpy()[0]
    sr_feat = sr_feat.transpose(0, 2, 3, 1)
    N,C,H,W = sr_feat.shape
    # for i in range(N):
    #     cv2.imwrite(os.path.join(dir, 'cam_{}.jpg'.format(i)), np.uint8(255 * sr_feat[i]))

    heatmap = F.interpolate(heatmap, size=[128, 128], mode="bilinear")

    # img_lr = F.interpolate(heatmap, size=[128, 128], mode="bilinear")
    heatmaps = heatmap.cpu().numpy()


    img = img.cpu()
    pic = img.numpy()[0]
    img = img.permute(0,2,3,1)

    heatmap = heatmaps[0]
    C,H,W = heatmap.shape
    heatmap_all = heatmap.sum(0)
    img = img[0]

    for i in range(C): # 20heatmap
        heatmap_i = cv2.applyColorMap(np.uint8(255 * heatmap[i]), cv2.COLORMAP_JET)
        heatmap_i = np.float32(heatmap_i) / 255
        heatmap_i = heatmap_i/np.max(heatmap_i)


        # cam = np.float32(img)
        # cam = cam + heatmap_i
        cam = heatmap_i
        cam = cam / np.max(cam)

        cv2.imwrite(os.path.join(dir,name+'_dir{}.jpg'.format(i)), np.uint8(255 * cam))

        cam = cam[:, :, ::-1]
        plt.figure(figsize=(10, 10))
        plt.imshow(np.uint8(255 * cam))


    # for i in range(C):   # 3heatmap  front,back,side
    #     heatmap_i = cv2.applyColorMap(np.uint8(255 * heatmap[i]), cv2.COLORMAP_JET)
    #     heatmap_i = np.float32(heatmap_i) / 255
    #     heatmap_i = heatmap_i/np.max(heatmap_i)
    #
    #     pic_i = cv2.applyColorMap(np.uint8(255 * pic[i]), cv2.COLORMAP_JET)
    #     pic_i = np.float32(pic_i) / 255
    #     pic_i = pic_i / np.max(pic_i)
    #
    #     image = pic_i
    #     image = image / np.max(image)
    #     # cam = np.float32(img)
    #     # cam = cam + heatmap_i
    #     cam = heatmap_i
    #     cam = cam / np.max(cam)
    #     if i==0:
    #        cv2.imwrite(os.path.join(dir,'cam_'+name+'front.jpg'), np.uint8(255 * cam))
    #        cv2.imwrite(os.path.join(dir, 'image_' + name + 'r.jpg'), np.uint8(255 * image))
    #     elif i==1:
    #         cv2.imwrite(os.path.join(dir,'cam_'+name+'back.jpg'), np.uint8(255 * cam))
    #         cv2.imwrite(os.path.join(dir, 'image_' + name + 'g.jpg'), np.uint8(255 * image))
    #     else:
    #         cv2.imwrite(os.path.join(dir, 'cam_' + name + 'side.jpg'), np.uint8(255 * cam))
    #         cv2.imwrite(os.path.join(dir, 'image_' + name + 'b.jpg'), np.uint8(255 * image))
    #     cam = cam[:, :, ::-1]
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(np.uint8(255 * cam))

    heatmap_all = cv2.applyColorMap(np.uint8(255*heatmap_all),cv2.COLORMAP_JET)
    heatmap_all = np.float32(heatmap_all)/255
    heatmap_all = heatmap_all / np.max(heatmap_all)

    cam = np.float32(img)
    cam = cam + heatmap_all
    cam = cam / np.max(cam)
    cv2.imwrite(os.path.join(dir, 'attention_'+name+'.png'),np.uint8(255*cam))
    cam = cam[:,:,::-1]
    plt.figure(figsize=(10,10))
    plt.imshow(np.uint8(255*cam))

def showPointSingleModal(features, label, save_path):
    # label = self.relabel(label)
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    features_tsne = tsne.fit_transform(features.cpu())
    COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black', 'blue','pink','yellow','green']
    MARKS = ['x', 'o', '+', '^', 's','D','d','1','8']
    features_min, features_max = features_tsne.min(0), features_tsne.max(0)
    features_norm = (features_tsne - features_min) / (features_max - features_min)
    plt.figure(figsize=(20, 20))
    for i in range(features_norm.shape[0]):
        plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[label[i] % 10],
                  marker=MARKS[label[i] % 9])
        # plt.scatter(features_norm[i, 0], features_norm[i, 1], str(label[i]), color=COLORS[label[i] % 6],
        #             marker=MARKS[label[i] % 5])
        # plt.text(features_norm[i, 0], features_norm[i, 1], str(label[i]), color=COLORS[label[i] % 6],
        #           fontdict={'weight':'bold', 'size':9})
    plt.savefig(save_path)
    plt.show()
    plt.close()

def showclassifier(features, label, save_path):
        # label = self.relabel(label)
        tsne = TSNE(n_components=2, init='pca', random_state=501)
        features_tsne = tsne.fit_transform(features.cpu())
        COLORS = ['black', 'blue', 'pink', 'gold', 'green', 'darkorange', 'firebrick', 'teal', 'olivedrab', 'rosybrown', 'chocolate', 'indigo']
        MARKS = ['x', 'o']
        features_min, features_max = features_tsne.min(0), features_tsne.max(0)
        features_norm = (features_tsne - features_min) / (features_max - features_min)

        plt.figure(figsize=(20, 20))

        for i in range(features_norm.shape[0]):
            if i < 131:
                if label[i] == 2:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[0],
                                marker=MARKS[0])
                elif label[i] == 5:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[1],
                                marker=MARKS[0])
                elif label[i] == 6:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[2],
                                marker=MARKS[0])
                elif label[i] == 9:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[3],
                                marker=MARKS[0])
                elif label[i] == 14:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[4],
                                marker=MARKS[0])
                elif label[i] == 118:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[5],
                                marker=MARKS[0])
                elif label[i] == 134:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[6],
                                marker=MARKS[0])
                elif label[i] == 177:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[7],
                                marker=MARKS[0])
                elif label[i] == 192:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[8],
                                marker=MARKS[0])
                elif label[i] == 273:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[9],
                                marker=MARKS[0])
                elif label[i] == 310:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[10],
                                marker=MARKS[0])
                elif label[i] == 402:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[11],
                                marker=MARKS[0])
            else:
                if label[i] == 2:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[0],
                                marker=MARKS[1])
                elif label[i] == 5:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[1],
                                marker=MARKS[1])
                elif label[i] == 6:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[2],
                                marker=MARKS[1])
                elif label[i] == 9:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[3],
                                marker=MARKS[1])
                elif label[i] == 14:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[4],
                                marker=MARKS[1])
                elif label[i] == 118:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[5],
                                marker=MARKS[1])
                elif label[i] == 134:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[6],
                                marker=MARKS[1])
                elif label[i] == 177:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[7],
                                marker=MARKS[1])
                elif label[i] == 192:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[8],
                                marker=MARKS[1])
                elif label[i] == 273:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[9],
                                marker=MARKS[1])
                elif label[i] == 310:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[10],
                                marker=MARKS[1])
                elif label[i] == 402:
                    plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[11],
                                marker=MARKS[1])
            # plt.scatter(features_norm[i, 0], features_norm[i, 1], str(label[i]), color=COLORS[label[i] % 6],
            #             marker=MARKS[label[i] % 5])
            # plt.text(features_norm[i, 0], features_norm[i, 1], str(label[i]), color=COLORS[label[i] % 6],
            #           fontdict={'weight':'bold', 'size':9})
        plt.savefig(save_path)
        # plt.show()
        plt.close()

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = Image.open(path)
    im = im.resize((128, 128))
    im = np.array(im)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

########################################################################
# Visualize the rank result
def rank_visual(query_i, qf, ql, qc, qi, gf, gl, gc, gi):
    # 定义哪个query
    index = sort_img(qf[query_i], ql[query_i], qc[query_i], gf, gl, gc)
    try:  # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(qi[query_i][0], 'query')
        for i in range(10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            img_path = gi[index[i]][0]
            label = gl[index[i]]
            imshow(img_path)
            if label == ql[query_i]:
                ax.set_title('%d' % (i + 1), color='green')
            else:
                ax.set_title('%d' % (i + 1), color='red')
            # print(img_path)
    except RuntimeError:
        for i in range(10):
            img_path = gi[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    name = str(query_i)+'.jpg'
    save_path = r"E:\Dataset_test\firstwork\picture\rank_map\KG"
    path = os.path.join(save_path,name)
    fig.savefig(path)