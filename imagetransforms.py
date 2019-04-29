from toolz import pipe as p, identity

from matplotlib import colors
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
    return overChannels(tensor, perChannelNorm)

def perChannelNorm(tensor3, ch):
    ch = tensor3[ch, :, :].unsqueeze(0)
    mn = ch.mean()
    sd = ch.std()

    return TF.normalize(ch, [mn], [sd])

PerImageNorm = transforms.Lambda(perImageNorm)


def scale0to1(tensor):
    return overChannels(tensor, scaleChannel0to1)

def scaleChannel0to1(tensor3, ch):
    ch = tensor3[ch, :, :]
    ch = (ch - ch.min())/(ch.max() - ch.min())
    return ch

Scale0To1 = transforms.Lambda(scale0to1)

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

def sharpenRGB(tensor, alpha, sigma):
    hsv_tensor = rgbToHsvTensor(tensor)
    hsv_tensor[2, :, :] = .5
    #hsv_tensor[2, :, :] = sharpen(hsv_tensor[2, :, :], 
    #    alpha, 
    #    sigma)
    
    #neg_ixs = hsv_tensor[2, :, :] < 0
    #gt1_ixs = hsv_tensor[2, :, :] > 1
    #hsv_tensor[neg_ixs, :, :] = 0
    #hsv_tensor[gt1_ixs, :, :] = 1
    
    return hsvToRgbTensor(hsv_tensor)

def rgbToHsvTensor(tensor):
    return p(tensor, 
        tensorToData,
        colors.rgb_to_hsv,
        lambda _: _.transpose(2, 0, 1),
        torch.tensor)

def hsvToRgbTensor(tensor):
    return p(tensor,
        tensorToData, 
        colors.hsv_to_rgb,
        lambda _: _.transpose(2, 0, 1),
        torch.tensor,
        lambda _: expand(_, len(tensor.shape))
    )

def SharpenRGB(alpha, sigma):
    return transforms.Lambda(lambda _: sharpenRGB(_, alpha, sigma))


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


def average(tensor, size):
    kernel = np.ones(size, np.float32)/(size[0] * size[1])
    return convolve2d(tensor, kernel)

def Average(size):
    return transforms.Lambda(lambda _: average(_, size))


defaultMn = [0.485, 0.456, 0.406]
defaultSd = [0.229, 0.224, 0.225]
DefaultNormalizeRGB = transforms.Normalize(defaultMn, defaultSd)


def convolve2d(tensor, kernel):
    filter4 = p(kernel, torch.tensor, lambda _: expand(_, 4))

    return overChannels(tensor, 
        lambda t, ch_ix: convolveChannel(t, ch_ix, filter4))


def convolveChannel(tensor3, ch_ix, filter4):
    ch = tensor3[ch_ix, :, :]
    ch = expand(ch, 4)
    
    return torch.conv2d(ch, filter4, padding = 2).squeeze(0).squeeze(0)


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
        raise invalidDimsError(n_channels, 'channels')

RGB = transforms.Lambda(toRGB)


def randomResize(img, size):
    interp = np.random.randint(0, 3)
    return TF.resize(img, size, interpolation=interp)

RandomResize = transforms.Lambda(randomResize)


def invalidDimsError(n, dim_type):
    return ValueError(
        str(n) + ' invalid number of ' + dim_type
    )


def overChannels(tensor, fn, make3dim = True):
    if make3dim:
        tensor = expand(tensor.squeeze(), 3)

    n_channels = tensor.shape[0]
    ret = torch.stack(
        [fn(tensor, i) for i in range(0, n_channels)]
        , 0).squeeze()
    
    return expand(ret, len(tensor.shape))


#assumes len(tensor.shape) <= n
def expand(tensor, n):
    curr_n = len(tensor.shape)
    if n == curr_n:
        return tensor
    
    dims = [-1 if _ <= curr_n else 1 for _ in range(n, 0, -1)]
    return tensor.expand(*dims)