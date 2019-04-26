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

#data_dir = 'main_train_set/'
data_dir = 'scrap_data2000/'

def createDataLoaders(data_dir, data_augment = True):
    return trainModel.create_dataloaders(data_dir, data_augment=data_augment)


image_datasets, dataloaders, dataset_sizes = createDataLoaders(data_dir)
class_names = image_datasets['train'].classes
n_classes = len(class_names)

defaultModel = localResnet.ResNet([2, 2, 2, 2], n_classes, p=.2, in_channels = 32).to(device)

def runIt(log_params_verbose = False, model = defaultModel, model_state_f = None, 
        data_augment = True,
        lr = .01, lr_epoch_size = 25, lr_gamma = .1,
        num_epochs_per_run = 25,
        start_run = 0, num_runs = 50):
    if model_state_f is not None:
        model.load_state_dict(torch.load(model_state_f))
        model.eval()

    _, dataloaders, dataset_sizes = createDataLoaders(data_dir,
        data_augment=data_augment)

    (model, best_acc, lr) = dropoutRuns.trainTheModel(model = model, 
            cutoff_acc = .96,
            dataloaders = dataloaders, dataset_sizes = dataset_sizes,
            log_dir_base = 'runs/model_data',
            log_params_verbose = log_params_verbose,
            lr = lr, lr_epoch_size = lr_epoch_size, lr_gamma = lr_gamma,
            num_epochs_per_run = num_epochs_per_run,
            start_run = start_run, num_runs = num_runs) 

    torch.save(model.state_dict(), 'best_model.pt')
