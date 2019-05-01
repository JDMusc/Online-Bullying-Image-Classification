import os
from shutil import copy
from toolz import pipe as p


def create(src_dir = 'image_data', 
    dest_dir = 'image_data_bully_0_1/'):

    joinSrc = lambda *_: os.path.join(src_dir, *_)
    
    bully_classes = [c for c in os.listdir(c) 
        if p(c, joinSrc, os.path.isdir)
        and c is not 'nonbullying']

    bully_fs = [(c, os.path.join(src_dir, c, f))
        for c in bully_classes
        for f in p(c, joinSrc, os.listdir)]

    def mkDir(d):
        if not os.path.exists(d):
            os.makedirs(d)
    
    bully_dest_dir = os.path.join(dest_dir, 'bullying')
    mkDir(bully_dest_dir)

    for (c, f) in bully_fs:
        p(f, 
            os.path.basename,
            lambda _: os.path.join(bully_dest_dir, c + '_' + _),
            lambda _: copy(f, _)
        )

    non_bully_src_dir = joinSrc('nonbullying')
    non_bully_dest_dir = os.path.join(dest_dir, 'nonbullying')
    mkDir(non_bully_dest_dir)

    for f in p(non_bully_src_dir, os.listdir):
        f = os.path.join(non_bully_src_dir, f)
        p(f,
            os.path.basename,
            lambda _: os.path.join(non_bully_dest_dir, _),
            lambda _: copy(f, _)

        