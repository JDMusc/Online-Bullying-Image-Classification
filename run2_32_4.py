import os
from toolz import pipe as p

import numpy as np

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import dropoutRuns
import localResnet
import trainModel

device = torch.device("cuda")

data_dir = 'main_train_set/'
image_datasets, dataloaders, dataset_sizes = trainModel.create_dataloaders(data_dir)
class_names = image_datasets['train'].classes
n_classes = len(class_names)

defaultModel = localResnet.ResNet([2, 2, 2, 2], n_classes, p=.2, in_channels = 32).to(device)

def runIt(log_params = False, model = defaultModel, model_state_f = None, 
        lr = .01):
    if model_state_f is not None:
        model.load_state_dict(torch.load(model_state_f))
        model.eval()

    (model, best_acc) = dropoutRuns.trainTheModel(model = model, 
            start_run = 0, num_runs = 50, cutoff_acc = .96,
            log_dir_base = 'runs/best_model_data',
            log_params = log_params,
            lr = lr) 

    torch.save(model.state_dict(), 'best_model.pt')
