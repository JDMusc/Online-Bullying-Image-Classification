from toolz import pipe as p

import numpy as np
import os
import torch
from torchvision import datasets, transforms

import imagetransforms as it

def createTransformList(use_original = False, shift_positive = False):
    ShiftPositive = transforms.Lambda(lambda tensor: tensor + tensor.min().abs())
    Identity = transforms.Lambda(lambda _: _)

    transforms_list = [
    transforms.Grayscale(),
    transforms.Resize(240),
    transforms.CenterCrop(224) if use_original else it.TopCenterCrop,
    transforms.ToTensor(),
    transforms.Normalize([.5], [.5]) if use_original else it.PerImageNorm,
    ShiftPositive if shift_positive else Identity
    ]

    return transforms_list


def createDataTransforms(crop_size, resize=None, 
                           data_augment = True):
    resize = crop_size + 26 if resize is None else resize
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(crop_size),
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            it.TopCenterCrop,
            transforms.ToTensor(),
            it.PerImageNorm
        ]),
        'val': transforms.Compose(
            createTransformList()
        ),
    }

    if not data_augment:
        data_transforms['train'] = transforms.Compose(
            [transforms.RandomHorizontalFlip()] +
            createTransformList()
            )
    
    return data_transforms

def createDataloaders(data_dir, input_size=224, 
        data_augment = True,
        folders = dict(train='train', val = 'val')):
    xs = ['train', 'val']

    data_transforms = createDataTransforms(input_size, input_size,
                                            data_augment=data_augment)
    
    image_datasets = {x: p(data_dir, 
                           lambda _:os.path.join(_, folders[x]),
                           lambda _: datasets.ImageFolder(_, data_transforms[x])
                          )
                      for x in xs}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers = 4)
                   for x in xs}

    dataset_sizes = {x: len(image_datasets[x]) for x in xs}
    
    return image_datasets, dataloaders, dataset_sizes