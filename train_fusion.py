#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as udata
from pytorch_msssim import ssim as ssim_lossfun
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
# from tools import batch_PSNR
from torch.utils.tensorboard import SummaryWriter 
# writer = SummaryWriter('./log/train')
# tensorboard --logdir=log --port 8123

from my_transformer_fusion import MultiFocusTransNet
# from multifocus_dataset import MultifocusGrayDataset
from dataset_coco import CocoMultifocus, tensor2cvimage, cvimage2tensor, likely_resize_640_480_

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

EPOCH = 1000
BATCH_SIZE =4


# model = DeblurNet_SRCNN(CHANNEL_SIZE, KERNEL_SIZE, guassian_kernel)
# model = model.to(device)
# model.load_state_dict(torch.load('ckpts/fusion_model.ckpt'))

# criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.NLLLoss2d()
# criterion = torch.nn.L1Loss()
# criterion = torch.nn.BCELoss()
# criterion = torch.nn.L1Loss()

# optimizer = optim.SGD(model.parameters(), lr=0.000001, weight_decay=1e-8, momentum=0.9)

# train_set = Dataset(KERNEL_SIZE, 0.5, 0.8, True)
# train_loader = udata.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

# val_set = Dataset(KERNEL_SIZE, 0.5, 0.8, False)
# val_loader = udata.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)


# print('train size {}, val size {} '.format(len(train_set), len(val_set)))


def save_net(net, path):
    torch.save(net.state_dict(), path)

BEST_PSNR_SSIM = 0

def test(epoch, model, dataloader): #, criterion, writer):\
    global BEST_PSNR_SSIM
    avg_psnr = 0
    avg_ssim = 0

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    test_bar = tqdm(dataloader)
    for src, b0, b1, b2 in test_bar:
        # print(src.shape, b0.shape)
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
        # loss = criterion(ypred, train_label)
        # ssim_loss = ssim_lossfun( ypred, train_label, data_range=1, size_average=True) 
        # print('l1 {}.. ssim {}.. '.format(loss.item(), ssim_loss.item()))

        s = tensor2cvimage(src.squeeze(0)).reshape(hi, wi, channel)
        q = tensor2cvimage(b0.squeeze(0)).reshape(hi, wi, channel)
        w = tensor2cvimage(b1.squeeze(0)).reshape(hi, wi, channel)
        e = tensor2cvimage(b2.squeeze(0)).reshape(hi, wi, channel)

        y = tensor2cvimage(ypred.squeeze(0)).reshape(hi, wi, channel)

        sum1 = np.concatenate((s, q))
        sum2 = np.concatenate((w, e))
        sum = np.concatenate((sum1, sum2), axis=1)

        ps = psnr(y, s, data_range=255)
        sm = ssim(y, s, data_range=255, multichannel=True)
        # print('psnr {}.. ssim {}.. '.format(ps, sm))
        avg_psnr += ps
        avg_ssim += sm
        test_bar.set_description('psnr {:.5f}.. ssim {:.5f}.. '.format(ps, sm))
 

    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)
    avg_psnr_ssim = 3 * avg_psnr + 100 * avg_ssim

    if avg_psnr_ssim > BEST_PSNR_SSIM:

        BEST_PSNR_SSIM = avg_psnr_ssim
        save_net(model, "ckpts/fusion_{}_PS_SM({:.3f}_{:.3f}).model".format(epoch, avg_psnr, avg_ssim))
        print('val ## save net...epoch:{} '.format(epoch))

step = 0

def train(epoch, model, dataloader, criterion, optimizer, writer):
    model.train()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)

    running_loss = 0.0
    total = 0
    global step
    train_bar = tqdm(dataloader) # python进度条库
    for src, b0, b1, b2 in train_bar:
        optimizer.zero_grad()
        y = []
        for i in [0,1,2]:
            # input = [b0[:,i,:,:], b1[:,i,:,:], b2[:,i,:,:]]
            input = torch.cat((b0[:,i:i+1,:,:], b1[:,i:i+1,:,:], b2[:,i:i+1,:,:]), dim = 1)
            input = input.to(device)
            
            yi = model(input)
            y.append(yi)

        ypred = torch.cat(y, dim = 1)

        train_label = src.to(device)
        loss = criterion(ypred, train_label)
        ssim_loss = ssim_lossfun( ypred, train_label, data_range=1, size_average=True) #

        loss_total = 5 * loss + (1-ssim_loss)* 1
        loss_total.backward()
        optimizer.step()

        total += src.size(0)
        running_loss += loss_total.item()

        # if total // 10 == 0:
        writer.add_scalar('loss/loss_train-ssim', ssim_loss.item(), step)
        writer.add_scalar('loss/loss_train-l1', loss.item(), step)
        step += 1
        train_bar.set_description('train EPOCH %d loss_avg: %.5f, loss-ssim : %.5f loss-l1 : %.5f ' % (epoch, running_loss/total, ssim_loss.item(), loss.item()))

    # return running_loss / total


def train_task_multi_fusion():
    writer = SummaryWriter('./log/train_multi_fusion')
    # train_set = MultifocusGrayDataset(h5path = "/home/csh/code/localproject/pytorch_networks/dataset6/multi_focus/multi_focus.h5")
    custom_dataset = CocoMultifocus()
    BATCH_SIZE = 1
    TRAIN_SIZE = int(len(custom_dataset) * 0.98)
    TEST_SIZE = len(custom_dataset) - TRAIN_SIZE # int(len(custom_dataset) * 0.1)

    # 指定随机种子， 每次都固定分配
    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [TRAIN_SIZE, TEST_SIZE], torch.Generator().manual_seed(0))
    train_loader = udata.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = udata.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    print(" ==== Dataset ==== ")
    print("train data size :{} test data size :{}".format(len(train_dataset), len(test_dataset)))
    print("train batch_size {}".format(BATCH_SIZE))
    print("input shape :{}, output shape :{}".format(train_dataset[0][0].shape, train_dataset[0][1].shape))
    print(" ==== Dataset ==== ")

    # val_set1 = Dataset_super([(5, 0.8), (7, 1.1), (9, 1.4)], True)
    # val_set2 = Dataset_super([(7, 0.8), (9, 1.1), (11, 1.4)], True)
    # val_set_union = MultiDataset([val_set1, val_set2])
    # val_loader = udata.DataLoader(val_set_union, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    model = MultiFocusTransNet((1, 3, 480, 640), patch_sizes_x=[40, 8 ,10], patch_sizes_y=[40, 8 ,10])
                    
    # t1 = torch.rand(1, 3, 560, 840)

    epoch_start = 0

    model = model.to(device)
    # model.load_state_dict(torch.load('ckpts/fusion_{}.model'.format(epoch_start)))

    criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0000005, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=1e-8)
    
    for epoch in range(EPOCH):
        if epoch <= epoch_start:
            # epoch += 1
            continue
        train(epoch, model, train_loader, criterion, optimizer, writer)
        test(epoch, model, test_loader) #, criterion, writer)
        # if epoch % 20 == 0:
        #     save_net(model, "ckpts/fusion_{}.model".format(epoch))

if __name__ == '__main__':
    train_task_multi_fusion()
