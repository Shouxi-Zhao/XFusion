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

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from net3 import MultiFocusTransNet
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


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

GPU_DEVICES = [1]

def test_fusion_on_dataset():

    # custom_dataset = CocoMultifocus()
    BATCH_SIZE = 2
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

    model = MultiFocusTransNet() #(1, 3, 480, 640), patch_sizes_x=[40, 8 ,10], patch_sizes_y=[40, 8 ,10])

    epoch_start = 325

    model=torch.nn.DataParallel(model,device_ids=GPU_DEVICES)
    model.cuda(GPU_DEVICES[0])
    # model = model.to(device)
    # model.load_state_dict(torch.load('ckpts/fusion_{}.model'.format(epoch_start)))
    model.load_state_dict(torch.load('ckpts/fusion_868_PS_SM(26.266_0.847).model'))

    criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    model.eval()

    avg_psnr = 0
    avg_ssim = 0
    num = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for src, b0, b1, b2 in test_bar:
            print(src.shape, b0.shape)
            b, channel, hi, wi = src.shape[0:4]
            ys = []
            for i in [0,1,2]:
                input = [b0[:,i:i+1,:,:], b1[:,i:i+1,:,:], b2[:,i:i+1,:,:]]
                input = torch.cat(input, dim = 1)
                input = input.cuda(device=GPU_DEVICES[0])
                
                yi = model(input)
                ys.append(yi)

            ypred = torch.cat(ys, dim = 1)

            train_label = src.cuda(device=GPU_DEVICES[0])
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

def test_fusion_single_img(path0, path1, path2, path3):

    im0 = cv2.imread(path0)
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)
    im3 = cv2.imread(path3)

    h,w = im1.shape[0:2]
    assert h == im2.shape[0] and h == im3.shape[0] and w == im2.shape[1] and w ==im3.shape[1], 'shape not fit'
    im0, im1, im2, im3 = likely_resize_640_480_(im0, im1, im2, im3) # np.zeros_like(im1, dtype=np.uint8), im1, im2, im3)
    im1, im2, im3 = cvimage2tensor(im1).unsqueeze(0), cvimage2tensor(im2).unsqueeze(0), cvimage2tensor(im3).unsqueeze(0)

    model = MultiFocusTransNet() #(1, 3, 480, 640), patch_sizes_x=[40, 8 ,10], patch_sizes_y=[40, 8 ,10])
    epoch_start = 169
    model=torch.nn.DataParallel(model,device_ids=GPU_DEVICES)
    model.cuda(GPU_DEVICES[0])
    # model = model.to(device)
    model.load_state_dict(torch.load('ckpts/fusion_{}.model'.format(epoch_start)))
    model.eval()

    with torch.no_grad():
        b, channel, hi, wi = im1.shape[0:4]
        y = []
        for i in [0,1,2]:
            input = [im1[:,i:i+1,:,:], im2[:,i:i+1,:,:], im3[:,i:i+1,:,:]]
            input = torch.cat(input, dim = 1)
            input = input.cuda(device=GPU_DEVICES[0])
            
            yi = model(input)
            y.append(yi)

        ypred = torch.cat(y, dim = 1)

        for bi in range(b):
            s = tensor2cvimage(tensor_unnormalize(ypred[bi])).reshape(hi, wi, channel)
            q = tensor2cvimage(tensor_unnormalize(im1[bi])).reshape(hi, wi, channel)
            w = tensor2cvimage(tensor_unnormalize(im2[bi])).reshape(hi, wi, channel)
            e = tensor2cvimage(tensor_unnormalize(im3[bi])).reshape(hi, wi, channel)

            ps = psnr(im0, s, data_range=255)
            sm = ssim(im0, s, data_range=255, multichannel=True)
            print('psnr {}.. ssim {}.. '.format(ps, sm))

            sum1 = np.concatenate((s, q))
            sum2 = np.concatenate((w, e))
            sum = np.concatenate((sum1, sum2), axis=1)

            cv2.imshow('sum', sum)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# img smaller than 480*640 
# return an img with original size
def test_fusion_single_img_ex(model, im0, im1, im2, im3):

    h,w = im1.shape[0:2]
    assert h == im2.shape[0] and h == im3.shape[0] and w == im2.shape[1] and w ==im3.shape[1], 'shape not fit'
    assert h <= 480  or w <= 640 , 'shape bigger 480 x 640 {}'.format(im1.shape)
    # print('input shape ', im0.shape)

    im0, im1, im2, im3 = likely_resize_640_480_(im0, im1, im2, im3) # np.zeros_like(im1, dtype=np.uint8), im1, im2, im3)
    im1, im2, im3 = cvimage2tensor(im1).unsqueeze(0), cvimage2tensor(im2).unsqueeze(0), cvimage2tensor(im3).unsqueeze(0)

    # print('in model shape ', im1.shape)
    with torch.no_grad():
        b, channel, hi, wi = im1.shape[0:4]
        y = []
        for i in [0,1,2]:
            input = [im1[:,i:i+1,:,:], im2[:,i:i+1,:,:], im3[:,i:i+1,:,:]]
            input = torch.cat(input, dim = 1)
            input = input.cuda(device=GPU_DEVICES[0])
            
            yi = model(input)
            y.append(yi)

        ypred = torch.cat(y, dim = 1)

        for bi in range(b):
            s = tensor2cvimage(tensor_unnormalize(ypred[bi])).reshape(hi, wi, channel)
            # q = tensor2cvimage(tensor_unnormalize(im1[bi])).reshape(hi, wi, channel)
            # ww = tensor2cvimage(tensor_unnormalize(im2[bi])).reshape(hi, wi, channel)
            # e = tensor2cvimage(tensor_unnormalize(im3[bi])).reshape(hi, wi, channel)

            # print('out model shape', s.shape)
            h_start = int((480-h)/2)
            w_start = int((640-w)/2)
            out = s[h_start:h_start+h , w_start:w_start+w,:]
            # print('out shape', out.shape)

            # sum1 = np.concatenate((s, q))
            # sum2 = np.concatenate((ww, e))
            # sum = np.concatenate((sum1, sum2), axis=1)
            # cv2.imshow('sum', sum)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return out

def test_fusion_img_list():
    import glob
    
    # dataDir = '/home/csh/code/localproject/pytorch_networks/dataset1/choosed_imgs/multi_focused/'
    # original_file_list = glob.glob(dataDir + "train2017_15_finished/original/*.jpg")
    original_file_list = valid_file_list

    for index in range(len(original_file_list)):
        file_path = original_file_list[index]

        defocus_path_1 = file_path.replace('original', 'multif').replace('.jpg', '_1.jpg')
        defocus_path_2 = file_path.replace('original', 'multif').replace('.jpg', '_2.jpg')
        defocus_path_3 = file_path.replace('original', 'multif').replace('.jpg', '_3.jpg')

        test_fusion_single_img(file_path, defocus_path_1, defocus_path_2, defocus_path_3)

def test_fusion_list_img_ex(path1_list, path2_list, path3_list = None, pathsave = './fusion_result_net3/',need_save = True, need_show = True):

    if need_save and not os.path.exists(pathsave):
        os.mkdir(pathsave)

    model = MultiFocusTransNet() #(1, 3, 480, 640), patch_sizes_x=[40, 8 ,10], patch_sizes_y=[40, 8 ,10])
    epoch_start = 169
    model=torch.nn.DataParallel(model,device_ids=GPU_DEVICES)
    model.cuda(GPU_DEVICES[0])
    # model = model.to(device)
    # model.load_state_dict(torch.load('ckpts/fusion_{}.model'.format(epoch_start)))

    checkpoint = torch.load('ckpts/fusion_2762.model', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    # model.load_state_dict(torch.load('ckpts/fusion_1950_PS_SM(26.676_0.860).model'))
    model.eval()

    with torch.no_grad():
        for index in range(len(path1_list)):
            im1 = cv2.imread(path1_list[index])
            im2 = cv2.imread(path2_list[index])
            # im3 = im1.copy() if path3_list is None else cv2.imread(path3_list[index])
            # im3 = cv2.GaussianBlur(im1, (3, 3), 0) if path3_list is None else cv2.imread(path3_list[index])
            # im3 = cv2.GaussianBlur(im1, (5, 5), 0) if path3_list is None else cv2.imread(path3_list[index])
            im3 = cv2.GaussianBlur(im1, (7, 7), 0) if path3_list is None else cv2.imread(path3_list[index])
            im3 = cv2.GaussianBlur(im1, (7, 7), 0) if path3_list is None else cv2.imread(path3_list[index])
            # im3 = cv2.GaussianBlur(im1, (13, 13), 0) if path3_list is None else cv2.imread(path3_list[index])
            im0 = np.zeros_like(im1, dtype=np.uint8)

            # enpand 10 px
            im1 = np.pad(im1, ((10,10),(10,10),(0,0)), 'constant', constant_values=255)
            im2 = np.pad(im2, ((10,10),(10,10),(0,0)), 'constant', constant_values=255)
            im3 = np.pad(im3, ((10,10),(10,10),(0,0)), 'constant', constant_values=255)
            im0 = np.pad(im0, ((10,10),(10,10),(0,0)), 'constant', constant_values=255)

            h,w = im1.shape[0:2]
            print(path1_list[index], '\r\n', path2_list[index])
            print('input shape ', im1.shape)
            assert im1.shape == im2.shape and im1.shape == im3.shape

            pred = np.zeros_like(im1, dtype=np.uint8)
            need_transpose = h > w
            if need_transpose:
                pred = pred.transpose(1,0,2)
                im0 = im0.transpose(1,0,2)
                im1 = im1.transpose(1,0,2)
                im2 = im2.transpose(1,0,2)
                im3 = im3.transpose(1,0,2)
                h , w = im0.shape[0:2]

            if h <= 480 and w <= 640:
                pred = test_fusion_single_img_ex(model, im0, im1, im2, im3)

            elif h > 480 and w <= 640:
                pred1 = test_fusion_single_img_ex(model, im0[:480,:,:], im1[:480,:,:], im2[:480,:,:], im3[:480,:,:])
                pred2 = test_fusion_single_img_ex(model, im0[-480:,:,:], im1[-480:,:,:], im2[-480:,:,:], im3[-480:,:,:])

                # pred1 = np.concatenate((pred1, pred2[480 - h:,:,:]))
                # pred2 = np.concatenate((pred1[:h-480,:,:], pred2))
                # pred = np.round(( pred1 + pred2 ) / 2)
                pred[-480:, :, :] = pred2
                pred[:470, :, : ] = pred1[:470,:,:]

            elif h <= 480 and w >640:
                pred1 = test_fusion_single_img_ex(model, im0[:,:640,:], im1[:,:640,:], im2[:,:640,:], im3[:,:640,:])
                pred2 = test_fusion_single_img_ex(model, im0[:,-640:,:], im1[:,-640:,:], im2[:,-640:,:], im3[:,-640:,:])

                # pred1 = np.concatenate((pred1, pred2[:,640-w:,:]), axis=1)
                # pred2 = np.concatenate((pred1[:,:w-640,:], pred2), axis=1)
                # pred = np.round(( pred1 + pred2 ) / 2)
                pred[:, -640:, :] = pred2
                pred[:, 0:630, : ] = pred1[:,0:630,:]

            elif h >480 and w >640:
                pred1 = test_fusion_single_img_ex(model, im0[:480, :640, :], im1[:480, :640, :], im2[:480, :640, :], im3[:480, :640, :])
                pred2 = test_fusion_single_img_ex(model, im0[:480, -640:, :], im1[:480, -640:, :], im2[:480, -640:, :], im3[:480, -640:, :])
                pred3 = test_fusion_single_img_ex(model, im0[-480:, :640, :], im1[-480:, :640, :], im2[-480:, :640, :], im3[-480:, :640, :])
                pred4 = test_fusion_single_img_ex(model, im0[-480:, -640:, :], im1[-480:, -640:, :], im2[-480:, -640:, :], im3[-480:, -640:, :])

                # sum1 = np.concatenate((pred1, pred2))
                # sum2 = np.concatenate((pred3, pred4))
                # sum = np.concatenate((sum1, sum2), axis=1)
                # cv2.imshow('sum', sum)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # pred[0:480, 0:640, :] = pred1
                # pred[0:480, 640:, :] = pred2[0:480:, 640 - w: , :]
                # pred[480 - h: , 0:640, :] = pred3[480 - h, 0:640, :]
                # pred[480-h:, 640-w:, :] = pred4[480-h:, 640-w:, :]

                pred[-480:, -640:, :] = pred4
                pred[-480:, 0:630, :] = pred3[:,0:630:,:]
                pred[0:470, -640:, :] = pred2[0:470,:,:]
                pred[0:470, 0:630, : ] = pred1[0:470,0:630,:]

                # pred1 = np.concatenate((pred1, pred2[:,640-w:,:]), axis=1)
                # pred2 = np.concatenate((pred1[:,:w-640,:], pred2), axis=1)
                # pred_up = np.round(( pred1 + pred2 ) / 2)
                # cv2.imshow('pred_up', pred1.astype(np.uint8))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # pred3 = np.concatenate((pred3, pred4[:,640-w:,:]), axis=1)
                # pred4 = np.concatenate((pred3[:,:w-640,:], pred4), axis=1)
                # pred_down = np.round(( pred3 + pred4 ) / 2)
                # cv2.imshow('pred_down', pred_down.astype(np.uint8))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # pred_up = np.concatenate((pred1, pred3[480 - h:,:,:]))
                # pred_down = np.concatenate((pred_up[:h-480,:,:], pred_down))
                # pred = np.round(( pred_up + pred_down ) / 2)
                # cv2.imshow('pred', pred.astype(np.uint8))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            else:
                print('gad !!!')

            print('result shape ', pred.shape)

            if need_transpose:
                im1 = im1.transpose(1,0,2)
                im2 = im2.transpose(1,0,2)
                im3 = im3.transpose(1,0,2)
                pred = pred.transpose(1,0,2)

            # deenpand 10 px
            pred = pred[10:-10,10:-10,:]

            if need_show:
                sum1 = np.concatenate((pred, im1))
                sum2 = np.concatenate((im2, im3))
                sum = np.concatenate((sum1, sum2), axis=1)
                cv2.imshow('sum', sum)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if need_save:
                name1 = path1_list[index].split('/')[-1].replace('_A','_result').replace('_B','_result')
                name2 = path2_list[index].split('/')[-1].replace('_A','_result').replace('_B','_result')

                cv2.imwrite(os.path.join(pathsave, name1), pred)
                    


def test_fusion_benchmark():
    import glob
    
    # dataDir = '/home/csh/code/localproject/pytorch_networks/dataset4/MFIF/input/'
    dataDir = '/home/user/ztr/pytorch_networks/multifocus_fusion/fusion_input/'
    original_file_list_1 = glob.glob(dataDir + "*/*_A.jpg")
    original_file_list_2 = glob.glob(dataDir + "*/*_B.jpg")

    original_file_list_1.sort()
    original_file_list_2.sort()

    test_fusion_list_img_ex(original_file_list_1, original_file_list_2, need_show=False)

if __name__ == '__main__':
    # test_fusion_on_dataset()
    # test_fusion_img_list()
    test_fusion_benchmark()
