import os
from toolz import pipe as p

import cv2
import pandas as pd

import utils

imageShape = lambda f: (cv2.imread(f)).shape


def imageShapesDir(im_dir):
    print('loading ' + im_dir)

    addDir = lambda f: os.path.join(im_dir, f)

    fs_dems = [(f, p(f, addDir, imageShape)) for f in os.listdir(im_dir) if
            utils.isImage(f) and utils.isOriginal(f)]
    
    fs, dems = utils.unzip(fs_dems)

    h, w, c = utils.unzip(dems)
    
    return p(
            dict(file=fs, dir = im_dir, height=h, width=w, channels = c), 
            pd.DataFrame)


def imageShapesDirs(im_dirs):
    return(pd.concat([imageShapesDir(d) for d in im_dirs]))


def writeImageShapesDirs(im_dirs):
    data = imageShapesDirs(im_dirs)
    data.to_csv('image_shapes.csv', index=False)

