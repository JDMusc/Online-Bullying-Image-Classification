from PIL import Image
import os
from shutil import copyfile
from toolz import pipe as p

import numpy as np
import pandas as pd
import torch

import modelEpochs

import preprocessing as pp

transform = pp.createDataTransforms(224)['val']

default_data_dir = 'image_data'


def loadDataset(data_dir):
    (dataset, _, _) = [x['val'] for x in pp.createDataloaders(
        data_dir, folders = dict(train='', val=''))]
    return dataset


def loadImg(f_name, device = torch.device('cuda')):
    img = p(f_name, Image.open, transform, lambda _: _.to(device))
    img.unsqueeze_(0)
    return img


def predict(model, data_dir, model_state_f = None):
    if model_state_f is not None:
        loadModel(model, model_state_f)
    
    def phasePreds(ph): 
        phase_preds = p(ph,
            lambda _: os.path.join(data_dir, _),
            lambda _: predictDir(model, model_data_dir=_)
            )
        
        phase_preds['phase'] = ph
        return phase_preds

    return p(['train', 'val'],
        lambda phs: [phasePreds(_) for _ in phs],
        pd.concat)


def predictFileProbs(model, f):
    return p(f, loadImg, model, torch.softmax)

def predictDir(model, model_state_f = None, model_data_dir = default_data_dir, analyze_data_dir = None):

    #assume analysis set has same structure as train set
    if analyze_data_dir is None:
        analyze_data_dir = model_data_dir

    analyze_dataset = loadDataset(analyze_data_dir)
    model_dataset = loadDataset(model_data_dir)

    if model_state_f is not None:
        loadModel(model, model_state_f)

    predIt = lambda f: p(f, loadImg, model, 
            lambda _: torch.max(_, 1)[1])

    preds = pd.DataFrame(analyze_dataset.samples, columns = ['file', 'class_ix'])
    preds['pred_ix'] = 0

    for i in range(0, preds.shape[0]):
        try:
            pred = predIt(preds.file[i])
            preds.loc[i, 'pred_ix'] = pred.cpu().numpy()
        except:
            preds.loc[i, 'pred_ix'] = -1


    def ixToClassFnGen(dataset):
        ix_to_class = {v:k for (k, v) in dataset.class_to_idx.items()}
        ix_to_class[-1] = 'NA'
        return lambda _: ix_to_class[_]

    label_classes = lambda ix_to_class_fn, col: [ix_to_class_fn(i) for i in preds[col]]
    preds['class'] = p(analyze_dataset, ixToClassFnGen, lambda _: label_classes(_, 'class_ix'))
    preds['pred_class'] = p(model_dataset, ixToClassFnGen, lambda _: label_classes(_, 'pred_ix'))

    return preds


def loadModel(model, model_state_f):
    model_state = p(model_state_f, torch.load)
    model.load_state_dict(model_state)
    model.eval()
    return model


def accuracy(preds, rows = None):
    if rows is not None:
        preds = preds.loc[rows,]

    return sum(preds.class_ix == preds.pred_ix)/preds.shape[0]


def performanceMetricsWithPhase(preds):
    return {_: performanceMetrics(preds.loc[preds.phase == _,]) for 
        _ in ['train', 'val']}


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


def mkdirIfNotExists(d):
    if not os.path.exists(d):
        os.mkdir(d)


def makeMisClassFolderWithPhase(preds_f, dest_dir):
    mkdirIfNotExists(dest_dir)
    
    preds = loadPreds(preds_f)
    for ph in ['train', 'val']:
        preds_ph = preds.loc[preds.phase == ph,]
        p(
            os.path.join(dest_dir, ph),
            lambda _: makeMisClassFolder(preds_ph, _)
        )


def makeMisClassFolder(preds, dest_dir):
    mkdirIfNotExists(dest_dir)

    preds = loadPreds(preds) if type(preds) is str else preds
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
