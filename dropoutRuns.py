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


default_data_dir = 'scrap_data2000/'
(default_image_datasets, default_dataloaders, 
        default_dataset_sizes) = trainModel.create_dataloaders(
                default_data_dir)
class_names = default_image_datasets['train'].classes
n_classes = len(class_names)

defaultModel = localResnet.ResNet([2, 2, 2, 2], n_classes, p=.2).to(device)

def runEpochs(model, i, 
        dataloaders, dataset_sizes,
        log_params_verbose, 
        lr, lr_epoch_size, lr_gamma,
        num_epochs,
        log_dir = None):

    log_dir = 'runs/dropout/' + str(i) if log_dir is None else log_dir

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_epoch_size, 
            gamma=lr_gamma)

    (model, best_acc) = trainModel.train_model(model, 
                                nn.CrossEntropyLoss(),
                                dataloaders, dataset_sizes, device,
                                log_params_verbose, num_epochs,
                                optimizer, scheduler,
                                writer = SummaryWriter(log_dir))
    return model, best_acc, scheduler.get_lr()[0]


def trainTheModel(model, 
        log_params_verbose,
        lr, lr_epoch_size, lr_gamma,
        num_epochs_per_run,
        cutoff_acc = None,
        dataloaders = default_dataloaders, 
        dataset_sizes = default_dataset_sizes,
        start_run=0, num_runs = 5,
        log_dir_base = 'runs/dropout/'):
    
    for i in range(start_run, start_run + num_runs):
        model, best_acc, lr = runEpochs(model, i = i,
                dataloaders = dataloaders, dataset_sizes = dataset_sizes,
                log_dir = log_dir_base + '_' + str(i),
                log_params_verbose = log_params_verbose,
                lr = lr, lr_epoch_size=lr_epoch_size, lr_gamma=lr_gamma,
                num_epochs = num_epochs_per_run)
        
        torch.save(model.state_dict(), 'model_' + str(i) + '.pt')

        if cutoff_acc is not None and best_acc > cutoff_acc:
            break;

    return model, best_acc, lr


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
