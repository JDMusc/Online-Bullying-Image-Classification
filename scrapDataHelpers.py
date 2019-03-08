import os
import shutil

import numpy as np

from toolz import pipe as p


def makeScrapData(classes, dest_dir = None, n_train = 30, n_val = None):
    if dest_dir is None:
        dest_dir = 'scrap_data' + str(n_train)

    fs = [os.path.join('image_data', c, f) 
          for c in classes
          for f in p(os.path.join('image_data', c), os.listdir)]

    fs = fs if n_val is None else np.random.choice(
            fs, n_train+n_val, replace = False)
    
    train_fs = np.random.choice(fs, n_train, replace = False)
    val_fs = [f for f in fs if f not in train_fs]
    
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
        
    os.mkdir(dest_dir)
    
    joinDirGen = lambda d: lambda f: os.path.join(d, f)
    joinScrapDir = joinDirGen(dest_dir)
    
    train_val_fs = dict(train=train_fs, val=val_fs)
    for tv in ('train', 'val'):
        p(tv, joinScrapDir, os.mkdir)
        
        for c in classes:
            p(c, joinDirGen(tv),  joinScrapDir, os.mkdir)    
            
        tv_fs = train_val_fs[tv]
        for f in tv_fs:
            f_class = ([c for c in classes if c in os.path.dirname(f)]).pop()
            dest = p(f,
                     os.path.basename,
                     joinDirGen(f_class), 
                     joinDirGen(tv),
                     joinScrapDir)
            shutil.copyfile(f, dest)
