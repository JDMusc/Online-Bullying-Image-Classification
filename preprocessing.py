from toolz import pipe as p

import numpy as np
import os
import torch
from torchvision import datasets, transforms

import imagetransforms as it

def baseTransformList(use_standard = False, random_resize = False):
    rz = transforms.Resize(240)
    if random_resize:
        rz = transforms.RandomResizedCrop(240, scale = (.4, 1.0))
    return [
        transforms.Grayscale(),
        rz,
        transforms.CenterCrop(224) if use_standard else it.TopCenterCrop(),
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5]) if use_standard else it.PerImageNorm
    ]


def augmentBaseTransforms(tforms, random_resize = False):
    tl = baseTransformList(random_resize=random_resize)
    return tl[0:-1] + tforms + [tl[-1]]


def createDataTransforms(crop_size, resize=None, 
                           data_augment = True):
    resize = crop_size + 26 if resize is None else resize
    data_transforms = {
        'train': transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            ] +
            augmentBaseTransforms(
                [transforms.RandomChoice(
                    [it.Sharpen(30, 1), it.Unsharpen, 
                        it.GaussianBlur(3)])],
                random_resize=True
            )
        ),
        'val': transforms.Compose(
            baseTransformList()
        )
    }

    if not data_augment:
        data_transforms['train'] = transforms.Compose(
            [transforms.RandomHorizontalFlip()] +
            baseTransformList()
            )
    
    return data_transforms

def createDataloaders(data_dir, input_size=224, 
        batch_size = 32,
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

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers = 4)
                   for x in xs}

    dataset_sizes = {x: len(image_datasets[x]) for x in xs}
    
    return image_datasets, dataloaders, dataset_sizes