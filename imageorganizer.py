import os
from PIL import Image
import shutil
from toolz import identity, pipe as p

import pandas as pd
from torchvision import transforms

import imagetransforms as it


def createImageModeTable():
    src_dir = 'image_data/'
    map_fn = lambda f: (f.replace(src_dir, ''), getMode(f))
    return p(src_dir, 
        lambda _: overClassFolders(_, map_fn=map_fn),
        lambda _: pd.DataFrame(_, columns = ['file', 'mode'])
    )


def createUniformDir(src_dir, dest_dir, mode):
    def map_fn(src_f):
        dest_f = src_f.replace(src_dir, dest_dir)

        dest_dir_c = os.path.dirname(dest_f)
        if not os.path.isdir(dest_dir_c):
            os.makedirs(dest_dir_c)

        shutil.copy(src_f, dest_f)

        return dest_f
    
    file_filter_fn = lambda _: getMode(_) == mode

    return overClassFolders(src_dir, map_fn=map_fn,
        file_filter_fn=file_filter_fn)


def findModeImages(src_dir, m):
    return overClassFolders(src_dir, 
        file_filter_fn = lambda _: getMode(_) == m)
        

def getMode(src_f):
    return Image.open(src_f).mode


def modeCounts(src_dir):
    ret = dict()

    def update_count(f):
        m = getMode(f)
        ret[m] = ret.get(m, 0) + 1

    overClassFolders(src_dir, map_fn = update_count)

    return ret    


def overClassFolders(src_dir, map_fn = identity, 
        file_filter_fn = lambda _: True):
    joinSrc = lambda *_: os.path.join(src_dir, *_)
    return [p(joinSrc(c, f), map_fn) for 
        c in os.listdir(src_dir) if p(c, joinSrc, os.path.isdir)
        for f in p(c, joinSrc, os.listdir) if p(joinSrc(c, f), file_filter_fn)
        ]


def writeImagesInColor(src_dir, dest_dir, n = None):
    img_transform = transforms.Compose(
        [transforms.ToTensor(),
        it.RGB,
        it.ToNumpy]
    )

    writeTransformedImages(src_dir, dest_dir, img_transform, n = n)