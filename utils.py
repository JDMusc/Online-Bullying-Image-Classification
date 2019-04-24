import os
from toolz import pipe as p

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
