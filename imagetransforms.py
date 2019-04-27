from toolz import pipe as p

import numpy as np
from scipy import ndimage
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


def imageToData(image, n_channels=3):
    convert_t = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return p(image, convert_t, tensorToData)


def gaussianBlur(arr, sigma):
    return ndimage.gaussian_filter(arr, sigma = sigma)

def GaussianBlur(sigma):
        return p(lambda _: gaussianBlur(_, sigma),
                numpyFnToTorchFn,
                transforms.Lambda)


def sharpen(arr, alpha, sigma):
    blurred = ndimage.gaussian_filter(arr, sigma)
    return arr + alpha * (arr - blurred)

def Sharpen(alpha, sigma):
    return p(
            lambda _: sharpen(_, alpha, sigma),
            numpyFnToTorchFn,
            transforms.Lambda
        )


def numpyFnToTorchFn(np_fn):
        return lambda tensor: p(tensor,
                tensorToData,
                np_fn,
                TF.to_tensor)


def unsharpen(arr):
    kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, -476, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], np.float32) / -256.0
    
    return ndimage.convolve(arr, kernel)

Unsharpen = p(unsharpen, numpyFnToTorchFn, transforms.Lambda)