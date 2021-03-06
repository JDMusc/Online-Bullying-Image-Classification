import os
from PIL import Image
import shutil
from toolz import pipe as p

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

import imagetransforms as it
import preprocessing as pp


def writeTransformedImage(src_f, dest_f, img_transform):
    dest_img = transformImage(src_f, img_transform)
    plt.imsave(dest_f, dest_img)

    #copy src file, easier to compare
    p(appendToName(dest_f, 'original'), 
        lambda _: shutil.copyfile(src_f, _))


def transformImage(src_f, img_transform):
    return p(src_f, Image.open, img_transform, it.ToNumpy, np.squeeze)


def writeTransformedImages(src_dir, dest_dir, img_transform,
        n = 10, scale0to1 = True):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    addRootGen = lambda r: lambda *_: os.path.join(r, *_)
    addSrc = addRootGen(src_dir)
    addDest = addRootGen(dest_dir)

    #facilitate viewing
    if scale0to1:
        img_transform.transforms.append(it.Scale0To1)

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
            dest_f = p(src_f, os.path.basename, fullDest)
            writeTransformedImage(src_f, dest_f, img_transform)


def appendToName(f_name, app):
    (f_name_pre_ext, ext) = os.path.splitext(f_name)
    return f_name_pre_ext + '_' + app + ext


def writeImagesInColor(src_dir, dest_dir, n = None):
    img_transform = transforms.Compose(
        [transforms.ToTensor(),
        it.RGB,
        it.ToNumpy]
    )

    writeTransformedImages(src_dir, dest_dir, img_transform, n = n)


def viewValTransformedImages(dest_dir, src_dir = 'image_data/', n = 10):
    img_transform = pp.createDataTransforms()['val']
    writeTransformedImages(src_dir, dest_dir, img_transform, n= n)


def viewTrainTransformedImages(dest_dir, src_dir = 'image_data/', n=10):
    img_transform = pp.createDataTransforms()['train']
    writeTransformedImages(src_dir, dest_dir, img_transform, n = n)