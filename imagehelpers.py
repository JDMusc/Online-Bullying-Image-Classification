import os
from PIL import Image
import shutil
from toolz import pipe as p

import numpy as np
from torchvision import transforms

def loadImage(img_f):
    return Image.open(img_f)


def writeTransformedImages(src_dir, dest_dir, transforms, n = 10):
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
        src_dir_fs = np.random.choice(src_dir_fs, n)

        for src_f in src_dir_fs:
            src_img = Image.open(src_f)
            dest_img = transforms(src_img)

            base_src = os.path.basename(src_f)
            dest_f = fullDest(base_src)
            dest_img.save(dest_f)

            #copy src file, easier to compare
            p(dest_f, 
                lambda _: appendToName(_, 'original'), 
                lambda _: shutil.copyfile(src_f, _))


def appendToName(f_name, app):
    (f_name_pre_ext, ext) = os.path.splitext(f_name)
    return f_name_pre_ext + '_' + app + ext
