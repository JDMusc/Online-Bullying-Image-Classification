import os
import shutil

import numpy as np

from toolz import pipe as p


def makeScrapData(classes, n_train = 30, n_val = 10):
    fs = [os.path.join('image_data', c, f) 
          for c in classes
          for f in p(os.path.join('image_data', c), os.listdir)]
    
    n_fs = n_train + n_val
    fs = np.random.choice(fs, n_fs, replace = False)
    
    train_fs = np.random.choice(fs, n_train, replace = False)
    val_fs = [f for f in fs if f not in train_fs]
    
    if os.path.exists('scrap_data'):
        shutil.rmtree('scrap_data')
        
    os.mkdir('scrap_data')
    
    joinDirGen = lambda d: lambda f: os.path.join(d, f)
    joinScrapDir = joinDirGen('scrap_data')
    
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