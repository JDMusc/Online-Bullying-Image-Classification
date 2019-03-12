import os
import shutil
from toolz import pipe as p

import numpy as np
import pandas as pd


def makeControlDir(dest_dir, keep_actions = None, drop_actions = None, n_total_images = 200, replace=True):
    
    dest_dir_exists = os.path.exists(dest_dir)
    if dest_dir_exists and replace:
        shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)
    elif not dest_dir_exists:
        os.mkdir(dest_dir)
    
    action_counts = _loadActionCounts(keep_actions, drop_actions, n_total_images)
    
    src_dir = 'stanford_40/JPEGImages'
    
    for c in action_counts.index:
        num_c = action_counts.loc[c, 'number_of_images']
        class_fs = np.random.choice(
            [f for f in os.listdir(src_dir) if c in f], num_c, replace = False)
        for f in class_fs:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))
    
    
def _loadActionCounts(keep_actions = None, drop_actions = None, n_total_images=200):
    if keep_actions is not None and drop_actions is not None:
        raise ValueError('can only chose keep actions or drop actions')
        
    f = "stanford_40/ImageSplits/actions.txt"
    action_counts = pd.read_csv(f, delim_whitespace=True, index_col = 0)
    
    actions = p(action_counts.index, set)
    
    if keep_actions is None and drop_actions is not None:
        drop_actions = drop_actions
    elif keep_actions is None and drop_actions is None:
        drop_actions = []
    else:
        keep_actions = [keep_actions] if type(keep_actions) is str else keep_actions
        drop_actions = actions - set(keep_actions)
    
    action_counts = action_counts.drop(drop_actions)
    action_counts['ratio'] = action_counts.number_of_images/sum(action_counts.number_of_images)
    action_counts['number_of_images_orig'] = action_counts.number_of_images
    action_counts['number_of_images'] = round(action_counts.ratio * n_total_images).astype(int)
    
    return action_counts
    
    