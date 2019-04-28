from toolz import pipe as p, identity

import numpy as np
from scipy import ndimage
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


def topCenterCrop(img, new_h = 224, new_w = 224):
    w = img.size[0]

    i = 0
    j = int( (w-new_w)/2 )
    return TF.crop(img, i, j, new_h, new_w)

def TopCenterCrop(new_h = 224, new_w = 224):
    return transforms.Lambda(
            lambda _: topCenterCrop(_, new_h = new_h, new_w = new_w)
        )


def perImageNorm(tensor):
    mn = [tensor.mean()]
    sd = [tensor.std()]
    return TF.normalize(tensor, mn, sd)

PerImageNorm = transforms.Lambda(perImageNorm)


def tensorToData(tensor):
    return tensor.numpy().transpose(1, 2, 0).squeeze()

ToNumpy = transforms.Lambda(tensorToData)


def imageToData(image, n_channels=3):
    convert_t = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return p(image, convert_t, tensorToData)


def gaussianBlur(tensor, sigma):
    kernel = p(centerOnlyMatrix(5),
        lambda _: ndimage.gaussian_filter(_, sigma = sigma)).astype(np.float32)
    return convolve2d(tensor, kernel)

def GaussianBlur(sigma):
    return transforms.Lambda(lambda _: gaussianBlur(_, sigma))


def sharpen(tensor, alpha, sigma):
    blurred = gaussianBlur(tensor, sigma)
    return tensor + alpha * (tensor - blurred)

def Sharpen(alpha, sigma):
    return transforms.Lambda(lambda _: sharpen(_, alpha, sigma))


def unsharpen(tensor):
    kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, -476, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], np.float32) / -256.0
    
    return convolve2d(tensor, kernel)

Unsharpen = transforms.Lambda(unsharpen)


def convolve2d(tensor, kernel):
    filter = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)

    n_dim = len(tensor.shape)
    n_unsqueeze = 4 - n_dim
    for _ in range(0, n_unsqueeze):
        tensor = tensor.unsqueeze(0)
    
    ret = torch.conv2d(tensor, filter, padding = 2)
    n_squeeze = len(ret.shape) - 3
    for _ in range(0, n_squeeze):
        ret = ret.squeeze(0)
    
    return ret


def centerOnlyMatrix(n):
    is_odd = n % 2 != 0
    if not is_odd:
        raise ValueError('n must be odd')
        
    m1 = np.zeros( (n,n))
    center = int(n/2)
    m1[center][center] = 1

    return m1


Identity = transforms.Lambda(identity)

def toRGB(tensor):
    n_channels = 1
    if len(tensor.shape) > 2:
        n_channels = tensor.shape[0]

    if n_channels == 3:
        return tensor
    elif n_channels == 1:
        return torch.stack([tensor, tensor, tensor], 1).squeeze()
    elif n_channels == 4:
        return tensor[0:3,:,:]
    else:
        raise ValueError(str(n_channels) + ' invalid number of channels')

RGB = transforms.Lambda(toRGB)