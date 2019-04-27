from toolz import pipe as p

import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF


def topCenterCrop(img, new_h = 224, new_w = 224):
    w = img.size[0]

    i = 0
    j = int( (w-new_w)/2 )
    return TF.crop(img, i, j, new_h, new_w)

TopCenterCrop = transforms.Lambda(topCenterCrop)


def perImageNorm(tensor):
    mn = [tensor.mean()]
    sd = [tensor.std()]
    return TF.normalize(tensor, mn, sd)

PerImageNorm = transforms.Lambda(perImageNorm)


def tensorToData(tensor):
    return tensor.numpy().transpose(1, 2, 0)


def imageToData(image, n_channels=3):
    convert_t = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return p(image, convert_t, tensorToData)


def createTransformList(use_original = False, shift_positive = False):
    ShiftPositive = transforms.Lambda(lambda tensor: tensor + tensor.min().abs())
    Identity = transforms.Lambda(lambda _: _)

    transforms_list = [
    transforms.Grayscale(),
    transforms.Resize(240),
    transforms.CenterCrop(224) if use_original else TopCenterCrop,
    transforms.ToTensor(),
    transforms.Normalize([.5], [.5]) if use_original else PerImageNorm,
    ShiftPositive if shift_positive else Identity
    ]

    return transforms_list

