from toolz import pipe as p
import PIL

import numpy as np
import os
import torch
from torchvision import datasets, transforms

import imagetransforms as it

SIZE = (224, 224)

def baseTransformList(random_resize = False):
    interp = PIL.Image.BILINEAR
    if random_resize:
        interp = p(range(0, 5), np.random.choice)

    return [
        it.ResizePIL(SIZE, interp),
        transforms.ToTensor(),
        it.RGB,
        it.PerImageNorm
    ]


def augmentBaseTransforms(tforms, start_ix, random_resize = False):
    tl = baseTransformList(random_resize=random_resize)
    return tl[0:start_ix] + tforms + tl[start_ix:]


def createDataTransforms(data_augment = True):
    data_transforms = {
        'train': transforms.Compose(
            augmentBaseTransforms(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter()]),
                    transforms.RandomChoice(
                        [
                            it.SharpenPIL,
                            it.UnsharpenPIL, 
                            it.GaussianBlurPIL(3),
                            it.Identity,
                            it.AvgBlurPIL(5)
                        ]
                    )
                ], start_ix = 1, random_resize=True)
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

    data_transforms = createDataTransforms(data_augment=data_augment)
    
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
