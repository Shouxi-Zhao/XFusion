#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 

import os
import numpy as np
from numpy.lib.function_base import delete
import torch
import torch.nn as nn
from torch import optim
from pytorch_msssim import ssim as ssim_lossfun
import torch.utils.data as udata
from tqdm import tqdm
# from tools import psnr,ssim

import h5py

import cv2

from my_transformer_fusion import MultiFocusTransNet
# from multifocus_dataset import MultifocusGrayDataset
from dataset_coco import CocoMultifocus, tensor2cvimage, cvimage2tensor, tensor_normalize, tensor_unnormalize, likely_resize_640_480_, train_file_list, valid_file_list


def tensor_to_np(tensor):
    print(tensor.size())
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def tensor_to_np_gray(tensor):
    img = torch.clamp(tensor.cpu(), min=0, max=1)
    img = torch.squeeze(img).detach().numpy()
    img = (img * 255).astype(np.uint8)

    # img = tensor.mul(255).byte()
    # img = img.cpu().numpy().squeeze(0)
    # # img = img *255
    # img[img>255] = 255
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

def test_fusion_on_dataset():

    # custom_dataset = CocoMultifocus()
    BATCH_SIZE = 1
    # TRAIN_SIZE = int(len(custom_dataset) * 0.95)
    # TEST_SIZE = len(custom_dataset) - TRAIN_SIZE # int(len(custom_dataset) * 0.1)

    test_dataset = CocoMultifocus(valid_file_list)
    # 指定随机种子， 每次都固定分配
    # train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [TRAIN_SIZE, TEST_SIZE], torch.Generator().manual_seed(0))
    # train_loader = udata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = udata.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    # print(" ==== Dataset ==== ")
    # print("train data size :{} test data size :{}".format(len(train_dataset), len(test_dataset)))
    # print("train batch_size {}".format(BATCH_SIZE))
    # print("input shape :{}, output shape :{}".format(train_dataset[0][0].shape, train_dataset[0][1].shape))
    # print(" ==== Dataset ==== ")

    model = MultiFocusTransNet((1, 3, 480, 640), patch_sizes_x=[40, 8 ,10], patch_sizes_y=[40, 8 ,10])

    epoch_start = 245

    model = model.to(device)
    model.load_state_dict(torch.load('ckpts/fusion_{}.model'.format(epoch_start)))

    criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    model.eval()

    avg_psnr = 0
    avg_ssim = 0
    num = 0

    test_bar = tqdm(test_loader)
    for src, b0, b1, b2 in test_bar:
        print(src.shape, b0.shape)
        b, channel, hi, wi = src.shape[0:4]
        ys = []
        for i in [0,1,2]:
            input = [b0[:,i:i+1,:,:], b1[:,i:i+1,:,:], b2[:,i:i+1,:,:]]
            input = torch.cat(input, dim = 1)
            input = input.to(device)
            
            yi = model(input)
            ys.append(yi)

        ypred = torch.cat(ys, dim = 1)

        train_label = src.to(device)
        loss = criterion(ypred, train_label)
        ssim_loss = ssim_lossfun( ypred, train_label, data_range=1, size_average=True) 
        print('l1 {}.. ssim {}.. '.format(loss.item(), ssim_loss.item()))


        num += b
        for bi in range(b):
            s = tensor2cvimage(tensor_unnormalize(src[bi])).reshape(hi, wi, channel)
            q = tensor2cvimage(tensor_unnormalize(b0[bi])).reshape(hi, wi, channel)
            w = tensor2cvimage(tensor_unnormalize(b1[bi])).reshape(hi, wi, channel)
            e = tensor2cvimage(tensor_unnormalize(b2[bi])).reshape(hi, wi, channel)

            y = tensor2cvimage(tensor_unnormalize(ypred[bi])).reshape(hi, wi, channel)

            sum1 = np.concatenate((s, q))
            sum2 = np.concatenate((w, e))
            sum = np.concatenate((sum1, sum2), axis=1)

            from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
            ps = psnr(y, s, data_range=255)
            sm = ssim(y, s, data_range=255, multichannel=True)
            print('psnr {}.. ssim {}.. '.format(ps, sm))

            avg_psnr += ps
            avg_ssim += sm

            cv2.imshow('sum', sum)
            cv2.imshow('y', y)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    avg_psnr /= num
    avg_ssim /= num
    print('imgs :{} avg_psnr:{} avg_ssim:{}'.format(num, avg_psnr, avg_ssim))

def test_fusion_single_img(path1, path2, path3):
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)
    im3 = cv2.imread(path3)

    h,w = im1.shape[0:2]
    assert h == im2.shape[0] and h == im3.shape[0] and w == im2.shape[1] and w ==im3.shape[1], 'shape not fit'
    im_null, im1, im2, im3 = likely_resize_640_480_(np.zeros_like(im1, dtype=np.uint8), im1, im2, im3)
    im1, im2, im3 = cvimage2tensor(im1).unsqueeze(0), cvimage2tensor(im2).unsqueeze(0), cvimage2tensor(im3).unsqueeze(0)

    model = MultiFocusTransNet((1, 3, 480, 640), patch_sizes_x=[40, 8 ,10], patch_sizes_y=[40, 8 ,10])
    epoch_start = 720
    model = model.to(device)
    model.load_state_dict(torch.load('ckpts/fusion_{}.model'.format(epoch_start)))
    model.eval()

    b, channel, hi, wi = im1.shape[0:4]
    y = []
    for i in [0,1,2]:
        input = [im1[:,i:i+1,:,:], im2[:,i:i+1,:,:], im3[:,i:i+1,:,:]]
        input = torch.cat(input, dim = 1)
        input = input.to(device)
        
        yi = model(input)
        y.append(yi)

    ypred = torch.cat(y, dim = 1)

    for bi in range(b):
        s = tensor2cvimage(tensor_unnormalize(ypred[bi])).reshape(hi, wi, channel)
        q = tensor2cvimage(tensor_unnormalize(im1[bi])).reshape(hi, wi, channel)
        w = tensor2cvimage(tensor_unnormalize(im2[bi])).reshape(hi, wi, channel)
        e = tensor2cvimage(tensor_unnormalize(im3[bi])).reshape(hi, wi, channel)

        sum1 = np.concatenate((s, q))
        sum2 = np.concatenate((w, e))
        sum = np.concatenate((sum1, sum2), axis=1)

        cv2.imshow('sum', sum)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_fusion_img_list():
    import glob
    
    dataDir = '/home/csh/code/localproject/pytorch_networks/dataset1/choosed_imgs/multi_focused/'
    original_file_list = glob.glob(dataDir + "train2017_5_finished/original/*.jpg")

    for index in range(len(original_file_list)):
        file_path = original_file_list[index]

        defocus_path_1 = file_path.replace('original', 'multif').replace('.jpg', '_1.jpg')
        defocus_path_2 = file_path.replace('original', 'multif').replace('.jpg', '_2.jpg')
        defocus_path_3 = file_path.replace('original', 'multif').replace('.jpg', '_3.jpg')

        test_fusion_single_img(defocus_path_1, defocus_path_2, defocus_path_3)


if __name__ == '__main__':
    test_fusion_on_dataset()
    # test_fusion_img_list()