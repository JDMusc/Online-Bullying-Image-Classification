import os
from PIL import Image
import shutil
from toolz import pipe as p

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from torchvision import transforms
import torchvision.transforms.functional as TF


def loadImage(img_f):
    return Image.open(img_f)


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


def createTransformList(use_original = False):

    transforms_list = [
        transforms.Grayscale(),
        transforms.Resize(240),
        transforms.CenterCrop(224) if use_original else TopCenterCrop,
        transforms.ToTensor(),
        transforms.Normalize([.5], [.5]) if use_original else PerImageNorm
    ]

    return transforms_list


def createAugmentTransformList():
    transforms_list = [
        transforms.Grayscale(),
        transforms.Resize(240),
        TopCenterCrop,
        transforms.ToTensor(),
        PerImageNorm
    ]

    return transforms_list


def writeTransformedImages(src_dir, dest_dir, 
        use_original = False, n = 10):
    img_transform = transforms.Compose(
        createTransformList(use_original=use_original)
    )

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    addRootGen = lambda r: lambda *_: os.path.join(r, *_)
    addSrc = addRootGen(src_dir)
    addDest = addRootGen(dest_dir)

    classes = [d for d in os.listdir(src_dir) if p(d, addSrc, os.path.isdir)]
    for c in classes:
        if not p(c, addDest, os.path.exists):
            p(c, addDest, os.mkdir)

        fullSrc = lambda f: addSrc(c, f)
        fullDest = lambda f: addDest(c, f)

        src_dir_fs = [fullSrc(f) for f in p(c, addSrc, os.listdir) 
            if p(f, fullSrc, os.path.isfile)]

        if n is not None:
            src_dir_fs = np.random.choice(src_dir_fs, n)

        for src_f in src_dir_fs:
            src_img = Image.open(src_f)
            dest_img = p(src_img, img_transform, np.squeeze)

            base_src = os.path.basename(src_f)
            dest_f = fullDest(base_src)
            plt.imsave(dest_f, dest_img, cmap="gray")

            #copy src file, easier to compare
            p(dest_f, 
                lambda _: appendToName(_, 'original'), 
                lambda _: shutil.copyfile(src_f, _))


def appendToName(f_name, app):
    (f_name_pre_ext, ext) = os.path.splitext(f_name)
    return f_name_pre_ext + '_' + app + ext
