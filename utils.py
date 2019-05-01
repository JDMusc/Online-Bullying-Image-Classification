import os
from toolz import pipe as p

import pandas as pd


def isImage(f):
    f = f.lower()
    return(any([e in f for e in ('jpg', 'jpeg', 'png')]))


def isOriginal(f):
    return(all([e not in f for e in ('flipped', 'noisy')]))


def unzip(tuple_arr):
    return p(zip(*tuple_arr), list)


def fileCount(parent_dir):
    add_d = lambda f: os.path.join(parent_dir, f)
    over_children = lambda fn: [c for c in os.listdir(parent_dir) if p(c, add_d, fn)]

    n_fs = p(os.path.isfile, over_children, len)
    ds = over_children(os.path.isdir)

    return n_fs + sum(p(d, add_d, fileCount) for d in ds)


def fileCountByClass(parent_dir):
    add_root = lambda *_: os.path.join(parent_dir, *_)
    classes = [c for c in os.listdir(parent_dir) if p(c, add_root, os.path.isdir)]
    return {c: p(c, add_root, fileCount) for c in classes}


def classFileCountsToCsv(class_file_counts, f_name = 'class_counts.csv'):
    pd.DataFrame(class_file_counts.keys(), class_file_counts.values()).to_csv(f_name)

