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

import trainModel

import localResnet

device = torch.device("cuda")


data_dir = 'scrap_data2000/'
image_datasets, dataloaders, dataset_sizes = trainModel.create_dataloaders(data_dir)
class_names = image_datasets['train'].classes
n_classes = len(class_names)

defaultModel = localResnet.ResNet([2, 2, 2, 2], n_classes, p=.2).to(device)

def runEpochs(model, i = 1, log_dir = None):

    log_dir = 'runs/dropout/' + str(i) if log_dir is None else log_dir

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=.9)
    (model, best_acc) = trainModel.train_model(model, nn.CrossEntropyLoss(),
                                   optimizer,
                                   lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
                                   dataloaders, dataset_sizes,
                                   device,
                                   writer = SummaryWriter(log_dir),
                                   num_epochs = 25
                                  )
    return model, best_acc


def trainTheModel(model = defaultModel, start_run=0, num_runs = 5,
        log_dir_base = 'runs/dropout/', cutoff_acc = None):
    
    for i in range(start_run, start_run + num_runs):
        model, best_acc = runEpochs(model, i, 
                log_dir = log_dir_base + '_' + str(i))

        if cutoff_acc is not None and best_acc > cutoff_acc:
            break;

    return model, best_acc


def tryCombos(num_runs):
    #ps = [None, .2, .5]
    ps = [.5]
    in_channels = [32, 64]
    block_sizes = [ [2, 2], [2, 2, 2], [2, 2, 2, 2] ]

    for p in ps:
        for in_channel in in_channels:
            for block_size in block_sizes:
                model_name = '_'.join(
                        [str(comp) for comp in (p, in_channel, len(block_size))])

                log_dir_base = 'runs/dropout' + '_' + model_name
                print(log_dir_base)

                model = localResnet.ResNet(
                        block_size, n_classes, p=p, 
                        in_channels=in_channel).to(device)

                (model, best_acc) = trainTheModel(model, num_runs = num_runs, 
                        log_dir_base=log_dir_base,
                        cutoff_acc = .96)

                torch.save(model.state_dict(), model_name + '.pt')
                
                np.savetxt(model_name + '.txt', [best_acc])
