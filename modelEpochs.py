import copy
import numpy as np
from numpy import log10
import os
from toolz import pipe as p

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn

import numpy as np

import preprocessing as pp


def findParam(model, name_filter):
    if callable(name_filter):
        fn = name_filter
    else:
        name_filter = [name_filter] if type(name_filter) is str else name_filter
        fn  = lambda param_name: all(
            component in param_name for component in name_filter)
        
    return [(pn, pv) for (pn, pv) in model.named_parameters() if fn(pn)]


def setParameterRequiresGrad(model, requires_grad = False, params = None):
    params = model.parameters() if params is None else params
    for param in params:
        param.requires_grad = requires_grad


def runEpochs(
    model, criterion, 
    dataloaders, dataset_sizes, device, 
    log_params_verbose, num_epochs,
    optimizer, scheduler,  
    writer):

    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    prev_model_wts = best_model_wts
    for epoch in range(num_epochs):
        epoch_acc, model_wts = _run_epoch(
            model, 
            criterion, dataloaders, dataset_sizes, device, 
            epoch, log_params_verbose, num_epochs, 
            optimizer, scheduler, writer)
        
        _log_coef_diffs(writer, epoch, prev_model_wts, model_wts)
        prev_model_wts = model_wts

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_wts

    # load best model weights
    model.load_state_dict(best_model_wts)
    return (model, best_acc)


def viewParamsToBeUpdated(model):
    return [n for (n,p) in model.named_parameters() if p.requires_grad == True]


def add_graph_model(writer, model, dataloaders, device):
    inputs, classes = p(dataloaders['train'], iter, next)
    
    inputs = inputs.to(device)
    classes = classes.to(device)
    
    writer.add_graph(model, inputs)


def _run_epoch(model, 
            criterion, dataloaders, dataset_sizes, device, 
            epoch, log_params_verbose, num_epochs,
            optimizer, scheduler, writer):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    n_samples = {'train': 0, 'val': 0}

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        is_train = phase == 'train'

        if is_train:
            scheduler.step()
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            n_samples[phase] = n_samples[phase] + len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)
            preds, loss =  _take_step(
                model, criterion, optimizer, inputs, labels, is_train)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        _log_epoch_phase_stats(writer, epoch, phase, epoch_loss, epoch_acc)

        if log_params_verbose:
            _log_model_params_verbose(writer, model, epoch, phase)

    # deep copy the model
    model_wts = copy.deepcopy(model.state_dict())

    _log_lr(writer, epoch, scheduler)
    print('# training samples')
    print(n_samples['train'])
    print('# val samples')
    print(n_samples['val'])
            
    return epoch_acc, model_wts



def _take_step(model, criterion, optimizer, inputs, labels, is_train):
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    # track history if only in train
    with torch.set_grad_enabled(is_train):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if is_train:
            loss.backward()
            optimizer.step()
    
    return preds, loss


def _add_scope(scope, k):
    return scope + '/' + k
    

def _add_scope_gen(scope):
    return lambda k: _add_scope(scope, k)


def _log_model_params_verbose(writer, model, run_num, scope, use_hist = False):
    def write(tag, param):
        fn = writer.add_histogram if use_hist else writer.add_scalar
        param = param if use_hist else param.abs().mean()
        return fn(tag, param, run_num)
    
    with torch.no_grad():
        for (name, param) in model.named_parameters():
            p(name, 
              _add_scope_gen(scope),
              lambda tag: write(tag, param)
             )


def _log_lr(writer, epoch, scheduler):
    lr = p(scheduler.get_lr(), np.array)[0]
    p('lr', 
        _add_scope_gen('lr'),
        lambda _: writer.add_scalar(_, lr, epoch)
        )
    p('log10_lr',
        _add_scope_gen('lr'),
        lambda _: writer.add_scalar(_, log10(lr), epoch)
        )


def _log_epoch_phase_stats(writer, epoch, scope, epoch_loss, epoch_acc):  

    log_measure = lambda k, v: p(k,
                                 _add_scope_gen(scope),
                                 lambda _ : writer.add_scalar(_, v, epoch)
                                )
    
    log_measure('loss', epoch_loss)
    log_measure('accuracy', epoch_acc)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        scope, epoch_loss, epoch_acc))
    

def _log_coef_diffs(writer, epoch, prev_model_state, curr_model_state):
    def write(name, curr):
        diff = curr - prev_model_state[name]
        p(name,
            _add_scope_gen('params'),
            lambda _: writer.add_scalar(
                _ + '.diff', diff.abs().mean(), epoch)
        )

    with torch.no_grad():
        for name in curr_model_state:
            if ('weight' in name or 'bias' in name): 
                write(name, curr_model_state[name])


