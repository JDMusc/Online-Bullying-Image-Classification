from PIL import Image
import os
from shutil import copyfile
from toolz import pipe as p

import numpy as np
import pandas as pd
import torch

import trainModel

device = torch.device('cuda')

transform = trainModel.create_data_transforms(224)['val']

default_data_dir = 'image_data'

def loadDataset(data_dir):
    (dataset, _, _) = [x['val'] for x in trainModel.create_dataloaders(
        data_dir, folders = dict(train='', val=''))]
    return dataset


def loadImg(f_name):
    img = p(f_name, Image.open, transform, lambda _: _.to(device))
    img.unsqueeze_(0)
    return img


def predict(model, model_state_f = None, data_dir = default_data_dir):
    dataset = loadDataset(data_dir)

    if model_state_f is not None:
        model_state = p(model_state_f, torch.load)
        model.load_state_dict(model_state)
        model.eval()

    predIt = lambda f: p(f, loadImg, model, 
            lambda _: torch.max(_, 1)[1])

    preds = pd.DataFrame(dataset.samples, columns = ['file', 'class_ix'])
    preds['pred_ix'] = 0

    for i in range(0, preds.shape[0]):
        try:
            pred = predIt(preds.file[i])
            preds.loc[i, 'pred_ix'] = pred.cpu().numpy()
        except:
            preds.loc[i, 'pred_ix'] = -1

    ix_to_class = {v:k for (k, v) in dataset.class_to_idx.items()}
    ix_to_class[-1] = 'NA'

    label_classes = lambda col: [ix_to_class[i] for i in ret[col]]
    preds['class'] = label_classes('class_ix')
    preds['pred_class'] = label_classes('pred_ix')

    return preds


def accuracy(preds, rows = None):
    if rows is not None:
        preds = preds.loc[rows,]

    return sum(preds.class_ix == preds.pred_ix)/preds.shape[0]


def performanceMetrics(preds):
    return {c: {
        'tpr': tpr(preds, c),
        'ppv': ppv(preds, c),
        'tnr': tnr(preds, c),
        'npv': npv(preds, c),
        'class_counts': classCounts(preds, c)
    } for c in np.unique(preds['class'])}


def ppv(preds, target):
    return accuracy(preds, rows = preds['pred_class'] == target)


def tpr(preds, target):
    return accuracy(preds, rows = preds['class'] == target)


def npv(preds, target):
    return accuracy(preds, rows = preds['pred_class'] != target)


def tnr(preds, target):
    return accuracy(preds, rows = preds['class'] != target)


def classCounts(preds, c):
    preds = preds.loc[preds['class'] == c,]

    pred_types = np.unique(preds['pred_class'])
    counts = [(c, sum(preds['pred_class'] == c)) for c in pred_types]
    counts.sort(key = lambda _: _[1], reverse = True)
    
    return counts


def makeMisClassFolder(preds_f, dest_dir):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    preds = loadPreds(preds_f)
    preds_mis = preds.loc[preds.class_ix != preds.pred_ix,]
    pred_classes = np.unique(preds_mis.pred_class)

    for c in pred_classes:
        c_dir = os.path.join(dest_dir, c)
        os.mkdir(c_dir)

        preds_mis_c = preds_mis.loc[preds_mis.pred_class == c,]
        for (_, r) in preds_mis_c.iterrows():
            p(r.file, 
                lambda _: r['class'] + '_' + os.path.basename(_),
                lambda _: os.path.join(c_dir, _),
                lambda _: copyfile(r.file, _)
            )


def loadPreds(preds_f):
    return pd.read_csv(preds_f, keep_default_na = False)