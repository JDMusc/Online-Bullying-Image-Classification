import copy
from toolz import pipe as p

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

import numpy as np


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def findParam(model, name_comps):
    name_comps = [name_comps] if type(name_comps) is str else name_comps
    return [(n, pa) for (n, pa) in model.named_parameters()
            if all(nc in n for nc in name_comps)]


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


defaultMn = [0.485, 0.456, 0.406]
defaultSd = [0.229, 0.224, 0.225]


def create_data_transforms(crop_size, resize=None, 
                           mn = defaultMn, sd = defaultSd):
    resize = crop_size + 26 if resize is None else resize
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(crop_size),
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mn, sd)
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mn, sd)
        ]),
    }
    return data_transforms


def train_model(
    model, criterion, optimizer, scheduler, dataloaders, 
    dataset_sizes, device, num_epochs=25):

    writer = SummaryWriter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_acc, model_wts = _run_epoch(
            model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
            device, num_epochs, epoch, writer)
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_wts

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def viewParamsToBeUpdated(model):
    return [n for (n,p) in model.named_parameters() if p.requires_grad == True]


def add_graph_model(writer, model, dataloaders, device):
    inputs, classes = p(dataloaders['train'], iter, next)
    
    inputs = inputs.to(device)
    classes = classes.to(device)
    
    writer.add_graph(model, inputs)


def _run_epoch(model, criterion, optimizer, scheduler, dataloaders, 
               dataset_sizes, device, num_epochs, epoch, writer):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        is_train = phase == 'train'
        is_val = not is_train

        if is_train:
            scheduler.step()
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds, loss =  _take_step(
                model, criterion, optimizer, inputs, labels, is_train)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        _log_epoch_stats(writer, epoch, phase, epoch_loss, epoch_acc)
        _log_model_params(writer, model, epoch, phase)

        # deep copy the model
    model_wts = copy.deepcopy(model.state_dict())
            
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


def _log_model_params(writer, model, run_num, scope, use_hist = False):
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


def _log_epoch_stats(writer, epoch, scope, epoch_loss, epoch_acc):  
    log_measure = lambda k, v: p(k,
                                 _add_scope_gen(scope),
                                 lambda _ : writer.add_scalar(_, v, epoch)
                                )
    
    log_measure('loss', epoch_loss)
    log_measure('accuracy', epoch_acc)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        scope, epoch_loss, epoch_acc))
    
