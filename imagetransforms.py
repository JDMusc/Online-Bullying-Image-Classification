import os
import re

import cv2
import numpy as np

import utils


def getId(img_f):
    return int(re.search('\d+', img_f).group())


def flipH(img):
    return cv2.flip(img, 1)


def addNoise(img, seed, sd_ratio=.1):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    l_channel = img_lab[:, :, 0]
    
    np.random.seed(seed)
    l_channel = np.random.normal(loc = 0, 
                             scale = l_channel.std() * sd_ratio, 
                             size = l_channel.shape) + l_channel
    l_channel[l_channel < 0] = 0
    l_channel[l_channel > 255] = 255
    
    img_lab[:, :, 0] = l_channel.astype(int)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


def interpIntoName(f_name, pattern):
    (f_name_no_ext, ext) = os.path.splitext(f_name)
    return f_name_no_ext + '_' + pattern + ext


def processFile(img_dir, f):
    full_f = os.path.join(img_dir, f)
    img = cv2.imread(full_f)
    
    flipped_img = flipH(img)
    
    f_id = getId(f)
    noisy_img = addNoise(img, f_id, .1)
    #add 1 so flipped does not have same noise
    noisy_flipped_img = addNoise(flipped_img, f_id+1, .1)

    cv2.imwrite(interpIntoName(full_f, 'noisy'), noisy_img)
    cv2.imwrite(interpIntoName(full_f, 'flipped'), flipped_img)
    cv2.imwrite(interpIntoName(full_f, 'noisy_flipped'), noisy_flipped_img)


def processDir(img_dir):
    fs = [f for f in os.listdir(img_dir) if utils.isImage(f) and 
          utils.isOriginal(f)]
    for f in fs:
        processFile(img_dir, f)
        
