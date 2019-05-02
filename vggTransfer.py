import torch
import torch.nn as nn
from torchvision import models

def loadVgg(n_classes = 9, device = "cuda"):
    device = torch.device(device)
    vgg = models.vgg19(pretrained=True).to(device) 

    for param in vgg.parameters():
        param.requires_grad = False

    n_inputs = 4096
    vgg.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(100, n_classes),
            nn.LogSoftmax(dim = 1))

    return vgg.to(device)


def viewBackPropParams(vgg):
    for (i, (n, param)) in enumerate(vgg.named_parameters()):
        if param.requires_grad:
            print(str(i) + ': ' + n)


def setParamGrad(vgg, param_ix, requires_grad):
    list(vgg.parameters())[param_ix].requires_grad = requires_grad


def unfreezeParam(vgg, param_ix):
    setParamGrad(vgg, param_ix, True)


def freezeParam(vgg, param_ix):
    setParamGrad(vgg, param_ix, False)


def unfreezeParams(vgg, param_ixs):
    for (i, p) in enumerate(vgg.parameters()):
        if i in param_ixs:
            p.requires_grad = True


def paramName(vgg, param_ix):
    return [n for (n, _) in vgg.named_parameters()][param_ix]


def paramNames(vgg, param_ixs):
    return [(i, paramName(vgg, i)) for i in param_ixs]


def paramIndex(vgg, param_name):
    return [i for (i, (n, _)) in enumerate(vgg.named_parameters()) if n == param_name][0]
