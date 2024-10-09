import os
import numpy as np
import torchvision
from torchvision.transforms.transforms import RandomVerticalFlip
# import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import h5py
import glob
import torch
import torch.nn.functional as F
import torch.utils.data as udata
import torchvision.transforms as transforms
import cv2
# from PIL import Image
import numpy as np



class ImageTools():

    @staticmethod
    def ycbcr_merge_c(c1, c2):
        assert c1.shape == c2.shape, "shape not equal {} {}".format(c1.shape, c2.shape)
        c_merge = np.ones_like(c1)

        H, W = c1.shape[0:2]

        for k in range(H):
            for n in range(W):
                if (abs(c1[k][n]-128)==0 and abs(c2[k][n]-128)==0):
                    c_merge[k][n] = 128
                else:
                    middle_1 = c1[k][n] * abs(c1[k][n]-128) + c2[k][n] * abs(c2[k][n]-128)
                    middle_2 = abs(c1[k][n]-128) + abs(c2[k][n]-128)
                    c_merge[k][n] = middle_1/middle_2
        return c_merge

    @staticmethod
    def ycbcr_merge_c3(c1, c2, c3):
        assert c1.shape == c2.shape and c1.shape == c3.shape, "shape not equal {} {}".format(c1.shape, c2.shape)
        c_merge = np.ones_like(c1)

        H, W = c1.shape[0:2]

        for k in range(H):
            for n in range(W):
                d1 = c1[k][n]
                d2 = c2[k][n]
                d3 = c3[k][n]
                if (d1 == 128 and d2 == 128 and d3 == 128):
                    c_merge[k][n] = 128
                else:
                    middle_1 = d1*abs(d1-128) + d2*abs(d2-128) + d3*abs(d3-128)
                    middle_2 = abs(d1-128) + abs(d2-128) + abs(d3-128)
                    # middle_1 = c1[k][n] * abs(c1[k][n]-128) + c2[k][n] * abs(c2[k][n]-128) + c3[k][n] * abs(c3[k][n]-128)
                    # middle_2 = abs(c1[k][n]-128) + abs(c2[k][n]-128) + abs(c3[k][n]-128)
                    c_merge[k][n] = middle_1/middle_2
        return c_merge
    
    @staticmethod
    def convert_rgb_to_ycbcr(img):
        if type(img) == np.ndarray:
            y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
            cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
            cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
            return np.array([y, cb, cr]).transpose([1, 2, 0])
        else:
            raise Exception('Unknown Type', type(img))

    @staticmethod
    def convert_ycbcr_to_rgb(img):
        if type(img) == np.ndarray:
            r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
            g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
            b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
            return np.array([r, g, b]).transpose([1, 2, 0])
        else:
            raise Exception('Unknown Type', type(img))



# https://blog.csdn.net/weixin_38208912/article/details/90297267

def cvimage2tensor(img):
    res = img / 255.
    res = res.transpose((2,0,1))
    res = torch.from_numpy(res).float()
    return res

def tensor2cvimage(tensor):
    img = torch.round(tensor.clamp(0, 1).mul(255)).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def tensor_normalize(tensor, mean=0.5, std=0.5):
    # return (tensor - mean) / std
    return tensor

def tensor_unnormalize(tensor, mean=0.5, std=0.5):
    # return tensor * std + mean
    return tensor
    
# input image size 640*425,640*426,640*480,640*479,447*640,
def likely_resize_640_480_(img, im1, im2, im3):

    assert img.shape == im1.shape and img.shape == im2.shape and img.shape == im3.shape

    h,w = img.shape[0:2]

    ps = h / w * 100
    sp = w / h * 100

    pad_l = 0
    pad_r = 0
    pad_u = 0
    pad_d = 0

    res_img = np.copy(img)

    if ps > 60 and ps < 82:  # 640 : 400~520
        if w >= 600 and w <= 640:
            if w == 640:
                pass
            else:
                p_l = int((640-w)/2)
                p_r = 640 - w -p_l
                # assert p_l > 0 and p_r >0 , "unfit shape {}".format(img.shape)
                pad_l = np.ones(shape = (h, p_l, 3), dtype = np.uint8) * 255
                pad_r = np.ones(shape = (h, p_r, 3), dtype = np.uint8) * 255

                res_img = np.concatenate((pad_l, res_img, pad_r), axis = 1)
                im1 = np.concatenate((pad_l, im1, pad_r), axis = 1)
                im2 = np.concatenate((pad_l, im2, pad_r), axis = 1)
                im3 = np.concatenate((pad_l, im3, pad_r), axis = 1)

            
            if h == 480:
                pass
            elif h < 480:
                p_u = int((480-h)/2)
                p_d = 480 - h - p_u
                # assert p_u > 0 and p_d >0 , "unfit shape .{}".format(img.shape)
                pad_u = np.ones(shape=(p_u, 640, 3), dtype=np.uint8) * 255
                pad_d = np.ones(shape=(p_d, 640, 3), dtype=np.uint8) * 255

                res_img = np.concatenate((pad_u, res_img, pad_d), axis = 0)
                im1 = np.concatenate((pad_u, im1, pad_d), axis = 0)
                im2 = np.concatenate((pad_u, im2, pad_d), axis = 0)
                im3 = np.concatenate((pad_u, im3, pad_d), axis = 0)
                
            else:
                p_u = int((h-480)/2)
                p_d = h - 480 - p_u
                res_img = res_img[p_u:p_u+480, :, :]
                im1 = im1[p_u:p_u+480, :, :]
                im2 = im2[p_u:p_u+480, :, :]
                im3 = im3[p_u:p_u+480, :, :]

        else:
            # print('unsupport shape (640 : 400~520):', cvimg.shape)
            # 500*375, 500*335, 500*333, 500*396, 
            if True: #w == 500:
                p_l = int((640-w)/2)
                p_r = 640 - w -p_l
                # assert p_l > 0 and p_r >0 , "unfit shape ..{}".format(img.shape)
                pad_l = np.ones(shape = (h, p_l, 3), dtype = np.uint8) * 255
                pad_r = np.ones(shape = (h, p_r, 3), dtype = np.uint8) * 255

                res_img = np.concatenate((pad_l, res_img, pad_r), axis = 1)
                im1 = np.concatenate((pad_l, im1, pad_r), axis = 1)
                im2 = np.concatenate((pad_l, im2, pad_r), axis = 1)
                im3 = np.concatenate((pad_l, im3, pad_r), axis = 1)

                p_u = int((480-h)/2)
                p_d = 480 - h - p_u
                # assert p_u > 0 and p_d >0 , "unfit shape ...{}".format(img.shape)
                pad_u = np.ones(shape=(p_u, 640, 3), dtype=np.uint8) * 255
                pad_d = np.ones(shape=(p_d, 640, 3), dtype=np.uint8) * 255

                res_img = np.concatenate((pad_u, res_img, pad_d), axis = 0)
                im1 = np.concatenate((pad_u, im1, pad_d), axis = 0)
                im2 = np.concatenate((pad_u, im2, pad_d), axis = 0)
                im3 = np.concatenate((pad_u, im3, pad_d), axis = 0)
            else:
                print('unsupport shape (640 : 400~520):', img.shape)
    elif sp > 60 and sp < 82: # 400~520 : 640

        return likely_resize_640_480_(img.transpose(1,0,2), im1.transpose(1,0,2), im2.transpose(1,0,2), im3.transpose(1,0,2))
    
    elif ps > 93 and ps < 107:
        # print('unsupport shape mid:', cvimg.shape)
        # 640*623 640*620 640*640 612*612 640*602 640*637
        if h > w:
            return likely_resize_640_480_(img.transpose(1,0,2), im1.transpose(1,0,2), im2.transpose(1,0,2), im3.transpose(1,0,2))
        else:
            if w < 640:
                p_l = int((640-w)/2)
                p_r = 640 - w -p_l
                assert p_l > 0 and p_r >0 , "unfit shape {}".format(img.shape)
                pad_l = np.ones(shape = (h, p_l, 3), dtype = np.uint8) * 255
                pad_r = np.ones(shape = (h, p_r, 3), dtype = np.uint8) * 255

                res_img = np.concatenate((pad_l, res_img, pad_r), axis = 1)
                im1 = np.concatenate((pad_l, im1, pad_r), axis = 1)
                im2 = np.concatenate((pad_l, im2, pad_r), axis = 1)
                im3 = np.concatenate((pad_l, im3, pad_r), axis = 1)

            if h == 480:
                pass
            elif h < 480:
                p_u = int((480-h)/2)
                p_d = 480 - h - p_u
                assert p_u > 0 and p_d >0 , "unfit shape .{}".format(img.shape)
                pad_u = np.ones(shape=(p_u, 640, 3), dtype=np.uint8) * 255
                pad_d = np.ones(shape=(p_d, 640, 3), dtype=np.uint8) * 255

                res_img = np.concatenate((pad_u, res_img, pad_d), axis = 0)
                im1 = np.concatenate((pad_u, im1, pad_d), axis = 0)
                im2 = np.concatenate((pad_u, im2, pad_d), axis = 0)
                im3 = np.concatenate((pad_u, im3, pad_d), axis = 0)
                
            else:
                p_u = int((h-480)/2)
                p_d = h - 480 - p_u
                res_img = res_img[p_u:p_u+480, :, :]
                im1 = im1[p_u:p_u+480, :, :]
                im2 = im2[p_u:p_u+480, :, :]
                im3 = im3[p_u:p_u+480, :, :]

    else:
        # print('unsupport shape:', cvimg.shape)
        # 640*290， 640*329~384， 640*541~569
        if h == 640:
            # print('unsupport shape:', img.shape)
            return likely_resize_640_480_(img.transpose(1,0,2), im1.transpose(1,0,2), im2.transpose(1,0,2), im3.transpose(1,0,2))
        elif w == 640:
            # print('unsupport shape:', img.shape)
            if h == 480:
                pass
            elif h < 480:
                p_u = int((480-h)/2)
                p_d = 480 - h - p_u
                assert p_u > 0 and p_d >0 , "unfit shape .{}".format(img.shape)
                pad_u = np.ones(shape=(p_u, 640, 3), dtype=np.uint8) * 255
                pad_d = np.ones(shape=(p_d, 640, 3), dtype=np.uint8) * 255

                res_img = np.concatenate((pad_u, res_img, pad_d), axis = 0)
                im1 = np.concatenate((pad_u, im1, pad_d), axis = 0)
                im2 = np.concatenate((pad_u, im2, pad_d), axis = 0)
                im3 = np.concatenate((pad_u, im3, pad_d), axis = 0)
            else:
                p_u = int((h-480)/2)
                p_d = h - 480 - p_u
                res_img = res_img[p_u:p_u+480, :, :]
                im1 = im1[p_u:p_u+480, :, :]
                im2 = im2[p_u:p_u+480, :, :]
                im3 = im3[p_u:p_u+480, :, :]
            
        else:
            # print('unsupport shape:', img.shape)
            # 500 * 281

            if h > w :
                return likely_resize_640_480_(img.transpose(1,0,2), im1.transpose(1,0,2), im2.transpose(1,0,2), im3.transpose(1,0,2))
            else:
                p_l = int((640-w)/2)
                p_r = 640 - w -p_l
                # assert p_l > 0 and p_r >0 , "unfit shape ..{}".format(img.shape)
                pad_l = np.ones(shape = (h, p_l, 3), dtype = np.uint8) * 255
                pad_r = np.ones(shape = (h, p_r, 3), dtype = np.uint8) * 255

                res_img = np.concatenate((pad_l, res_img, pad_r), axis = 1)
                im1 = np.concatenate((pad_l, im1, pad_r), axis = 1)
                im2 = np.concatenate((pad_l, im2, pad_r), axis = 1)
                im3 = np.concatenate((pad_l, im3, pad_r), axis = 1)

                p_u = int((480-h)/2)
                p_d = 480 - h - p_u
                # assert p_u > 0 and p_d >0 , "unfit shape ...{}".format(img.shape)
                pad_u = np.ones(shape=(p_u, 640, 3), dtype=np.uint8) * 255
                pad_d = np.ones(shape=(p_d, 640, 3), dtype=np.uint8) * 255

                res_img = np.concatenate((pad_u, res_img, pad_d), axis = 0)
                im1 = np.concatenate((pad_u, im1, pad_d), axis = 0)
                im2 = np.concatenate((pad_u, im2, pad_d), axis = 0)
                im3 = np.concatenate((pad_u, im3, pad_d), axis = 0)


    # print(res_img.shape, im1.shape, im2.shape)
    return res_img, im1, im2, im3



data_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    pass

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace: bool = False):
    """Unnormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def unnormalize_and_toPILimage(tensor):
    img = unnormalize(tensor).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def read_lines_from_file(file_path):
    res = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        keys=[k.strip('\n') for k in lines]
        return keys
        # for line in lines: 
        #     line = line.strip('\n')  #去掉列表中每一个元素的换行符
        #     res.append(line)
    print('path not exist', file_path)
    return None

def save_path_to_keyfile(file_list, key_path):

    with open(key_path,"w") as f:
        keys = []
        for file_path in file_list:
            # img = cv2.imread(file_path)
            key_ = file_path.split('/')
            key = '{}/{}/{}'.format(key_[-3], key_[-2] , key_[-1])
            keys.append(key)
        keys=[k+"\n" for k in keys]
        f.writelines(keys)
    f.close()
    print(len(keys))

def save_path_to_h5file(file_list, h5_path):
    h5f = h5py.File(h5_path, 'a')
    keys = []
    for file_path in file_list:
        img0 = cv2.imread(file_path)
        key_ = file_path.split('/')
        key = '{}/{}/{}'.format(key_[-3], key_[-2] , key_[-1])

        defocus_path_1 = file_path.replace('original', 'multif').replace('.jpg', '_1.jpg')
        defocus_path_2 = file_path.replace('original', 'multif').replace('.jpg', '_2.jpg')
        defocus_path_3 = file_path.replace('original', 'multif').replace('.jpg', '_3.jpg')

        key1 = key.replace('original', 'multif').replace('.jpg', '_1.jpg')
        key2 = key.replace('original', 'multif').replace('.jpg', '_2.jpg')
        key3 = key.replace('original', 'multif').replace('.jpg', '_3.jpg')

        img1 = cv2.imread(defocus_path_1)
        img2 = cv2.imread(defocus_path_2)
        img3 = cv2.imread(defocus_path_3)
        
        if img1 is None or img2 is None or img3 is None:
            print('error', key, file_path)

        img0, img1, img2, img3 = likely_resize_640_480_(img0, img1, img2, img3)

        h5f.create_dataset(str(key), data=img0)
        h5f.create_dataset(str(key1), data=img1)
        h5f.create_dataset(str(key2), data=img2)
        h5f.create_dataset(str(key3), data=img3)

        keys.append(key)
        print('save', key, file_path)
    h5f.close()
    print(keys)
    print(len(keys))



# dataDir = '/home/csh/code/localproject/pytorch_networks/dataset1/choosed_imgs/multi_focused/'
# original_file_list = glob.glob(dataDir + "train2017_*_finished/original/*.jpg")
# original_file_list.sort()

# valid_file_list = original_file_list[::19] # 跨步长19 取样
# # train_file_list = list(filter(lambda x: x % 20 != 0, original_file_list))
# train_file_list = list(set(original_file_list) - set(valid_file_list))


#dataDir = '/home/csh/code/localproject/pytorch_networks/multifocus_fusion/'
dataDir = '/home/user/ztr/pytorch_networks/multifocus_fusion/'
original_file_list = read_lines_from_file(dataDir + 'dataset_all.key')
valid_file_list = read_lines_from_file(dataDir + 'dataset_val.key')
train_file_list = list(set(original_file_list) - set(valid_file_list))

original_md5path = dataDir + 'dataset_all.h5py'



class CocoMultifocus(udata.Dataset):
    # mode = RGB/YCbCr/L=Gray
    def __init__(self, file_list = original_file_list, md5_path = original_md5path, re_size = 1.0, mode = 'RGB', transform = data_transforms):
        super(CocoMultifocus, self).__init__()

        self.file_list = file_list
        self.mode = mode
        self.data_transfrom = transform
        self.md5_path = md5_path

        self.re_size = re_size

        self.random_HorizontalFlip = True
        self.random_VerticalFlip = True
        self.random_order = True
        self.normal = False

    def loadImg(self, path):
        # return Image.open(path)
        # img = likely_resize_640_480_()
        # return Image.open(path).convert(self.mode)

        return cv2.imread(path)

    def loadImg_from_md5file(self, k):
        h5 = h5py.File(self.md5_path, 'r')
        img = np.array(h5[k])
        h5.close()
        return img

    def loadImg_from_md5file_4(self, k1,k2=None,k3=None,k4=None):
        h5 = h5py.File(self.md5_path, 'r')
        img1 = np.array(h5[k1])
        img2 = None if k2 is None else np.array(h5[k2])
        img3 = None if k3 is None else np.array(h5[k3])
        img4 = None if k4 is None else np.array(h5[k4])
        h5.close()
        return img1,img2,img3,img4

    def load_and_transfrom(self, path):
        self.data_transfrom(self.loadImg(path))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]

        defocus_path_1 = file_path.replace('original', 'multif').replace('.jpg', '_1.jpg')
        defocus_path_2 = file_path.replace('original', 'multif').replace('.jpg', '_2.jpg')
        defocus_path_3 = file_path.replace('original', 'multif').replace('.jpg', '_3.jpg')

        if self.md5_path is None:
            im0 = self.loadImg(file_path)
            im1 = self.loadImg(defocus_path_1)
            im2 = self.loadImg(defocus_path_2)
            im3 = self.loadImg(defocus_path_3)
            im0, im1, im2, im3 = likely_resize_640_480_(im0, im1, im2, im3)
        else:
            im0,im1,im2,im3 = self.loadImg_from_md5file_4(file_path, defocus_path_1, defocus_path_2, defocus_path_3)
        
        if self.re_size != 1.0:
            im0 = cv2.resize(im0,None,fx=self.re_size,fy=self.re_size,interpolation=cv2.INTER_CUBIC)
            im1 = cv2.resize(im1,None,fx=self.re_size,fy=self.re_size,interpolation=cv2.INTER_CUBIC)
            im2 = cv2.resize(im2,None,fx=self.re_size,fy=self.re_size,interpolation=cv2.INTER_CUBIC)
            im3 = cv2.resize(im3,None,fx=self.re_size,fy=self.re_size,interpolation=cv2.INTER_CUBIC)

        if self.random_HorizontalFlip and np.random.choice([True, False]):
            im0 = np.flip(im0, 1)
            im1 = np.flip(im1, 1)
            im2 = np.flip(im2, 1)
            im3 = np.flip(im3, 1)

        if self.random_VerticalFlip and np.random.choice([True, False]):
            im0 = np.flip(im0, 0)
            im1 = np.flip(im1, 0)
            im2 = np.flip(im2, 0)
            im3 = np.flip(im3, 0)

        if self.mode == "RGB":
            pass
        elif self.mode == "YCbCr":
            im0 = ImageTools.convert_rgb_to_ycbcr(im0)
            im1 = ImageTools.convert_rgb_to_ycbcr(im1)
            im2 = ImageTools.convert_rgb_to_ycbcr(im2)
            im3 = ImageTools.convert_rgb_to_ycbcr(im3)
        elif self.mode == "Gray":
            im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY)
            im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
            im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
            im3 = cv2.cvtColor(im3,cv2.COLOR_BGR2GRAY)
        

        # im0, im1, im2, im3 = likely_resize_640_480_(np.array(im0), np.array(im1), np.array(im2), np.array(im3))
        # im0 = self.data_transfrom(Image.fromarray(im0.astype('uint8')))
        # im1 = self.data_transfrom(Image.fromarray(im1.astype('uint8')))
        # im2 = self.data_transfrom(Image.fromarray(im2.astype('uint8')))
        # im3 = self.data_transfrom(Image.fromarray(im3.astype('uint8')))
        # return im0, im1, im2, im3

        # return cvimage2tensor(im0), cvimage2tensor(im1), cvimage2tensor(im2), cvimage2tensor(im3)
        if self.random_order:
            if self.normal:
                im0, im1, im2, im3 = tensor_normalize(cvimage2tensor(im0)), tensor_normalize(cvimage2tensor(im1)), \
                         tensor_normalize(cvimage2tensor(im2)), tensor_normalize(cvimage2tensor(im3))
            else:
                im0, im1, im2, im3 = cvimage2tensor(im0), cvimage2tensor(im1), cvimage2tensor(im2), cvimage2tensor(im3)
            res = [im1, im2, im3]
            random.shuffle(res)
            res.insert(0, im0)
            return tuple(res)
        else:
            if self.normal: 
                return tensor_normalize(cvimage2tensor(im0)), tensor_normalize(cvimage2tensor(im1)), \
                        tensor_normalize(cvimage2tensor(im2)), tensor_normalize(cvimage2tensor(im3))
            else:
                return cvimage2tensor(im0), cvimage2tensor(im1), cvimage2tensor(im2), cvimage2tensor(im3)

        # return F.normalize(cvimage2tensor(im0)), F.normalize(cvimage2tensor(im1)), \
        #     F.normalize(cvimage2tensor(im2)), F.normalize(cvimage2tensor(im3)) 



def test_dataset():

    d = CocoMultifocus(file_list= train_file_list, md5_path=original_md5path)
    print(len(d))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(84, 56))
    # plt.ion()

    # trans = transforms.ToPILImage(mode = 'RGB')
    for data in d:
        s, q, w ,e = data
        print(s.shape, q.shape, w.shape, e.shape)
        channel, hi, wi = s.shape[0:3]

        s = tensor2cvimage(tensor_unnormalize(s)).reshape(hi, wi, channel)
        q = tensor2cvimage(tensor_unnormalize(q)).reshape(hi, wi, channel)
        w = tensor2cvimage(tensor_unnormalize(w)).reshape(hi, wi, channel)
        e = tensor2cvimage(tensor_unnormalize(e)).reshape(hi, wi, channel)

        sum1 = np.concatenate((s, q))
        sum2 = np.concatenate((w, e))
        sum = np.concatenate((sum1, sum2), axis=1)

        cv2.imshow('sum', sum)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #     plt.subplot(221)
    #     plt.imshow(s)
    #     plt.title("s", fontsize=22)
    #     plt.axis('off')

    #     plt.subplot(222)
    #     plt.imshow(q)
    #     plt.title("b0", fontsize=22)
    #     plt.axis('off') 

    #     plt.subplot(223)
    #     plt.imshow(w)
    #     plt.title("b1", fontsize=22)
    #     plt.axis('off')

    #     plt.subplot(224)
    #     plt.imshow(e)
    #     plt.title("b2", fontsize=22)
    #     plt.axis('off') 

    #     plt.draw()
    #     plt.pause(3)

    # plt.ioff()
    # plt.show()

def test_1_():
    path = '/home/csh/code/localproject/pytorch_networks/dataset4/MFIF/input/MFI-WHU_16/MFI-WHU_16_A.jpg'
    im0 = cv2.imread(path)
    im0 = im0.transpose(1,0,2)
    print(im0.shape)
    res,_,_,_ = likely_resize_640_480_(im0, np.copy(im0), np.copy(im0), np.copy(im0))
    print(res.shape)

if __name__ == '__main__':
    test_dataset()
    # test_1_()
    # save_path_to_h5file(original_file_list, "dataset_all.h5py")
    # save_path_to_keyfile(valid_file_list, "dataset_val.key")
