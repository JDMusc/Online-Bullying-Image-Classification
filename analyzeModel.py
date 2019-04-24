from PIL import Image
from toolz import pipe as p

import pandas as pd
import torch

import trainModel

device = torch.device('cuda')

transform = trainModel.create_data_transforms(224)['val']

default_data_dir = 'image_data'

(default_dataset, _, _) = [x['val'] for x in trainModel.create_dataloaders(
    default_data_dir, folders = dict(train='', val=''))]


def loadImg(f_name):
    img = p(f_name, Image.open, transform, lambda _: _.to(device))
    img.unsqueeze_(0)
    return img


def analyze(model, model_state_f = None, data_dir = None):

    if data_dir is None:
        dataset = default_dataset
    else:
        (dataset, _, _) = [x['val'] for x in trainModel.create_dataloaders(
            data_dir, folders = dict(train='', val=''))]


    if model_state_f is not None:
        model_state = p(model_state_f, torch.load)
        model.load_state_dict(model_state)
        model.eval()

    predIt = lambda f: p(f, loadImg, model, 
            lambda _: torch.max(_, 1)[1])

    ret = pd.DataFrame(dataset.samples, columns = ['file', 'class_ix'])
    ret['pred_ix'] = 0

    for i in range(0, ret.shape[0]):
        try:
            pred = predIt(ret.file[i])
            ret.loc[i, 'pred_ix'] = pred.cpu().numpy()
        except:
            ret.loc[i, 'pred_ix'] = -1

    ix_to_class = {v:k for (k, v) in dataset.class_to_idx.items()}
    ix_to_class[-1] = 'NA'

    label_classes = lambda col: [ix_to_class[i] for i in ret[col]]
    ret['class'] = label_classes('class_ix')
    ret['pred_class'] = label_classes('pred_ix')

    return ret
