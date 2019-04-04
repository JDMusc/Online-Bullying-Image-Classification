#!/usr/bin/python

import sys

import io
from PIL import Image
from skimage import io
import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import localResnet

f_name = str(sys.argv[1])

print(f_name)

device = torch.device("cuda")
n_classes = 10
model = localResnet.ResNet([2, 2, 2], n_classes, in_channels=32).to(device)
model.load_state_dict(torch.load('model_final.pt'))
model.eval()


defaultMn = [0.485, 0.456, 0.406]
defaultSd = [0.229, 0.224, 0.225]

tform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(defaultMn, defaultSd)
])

classes = ['gossiping',
 'isolation',
 'laughing',
 'nonbullying',
 'pullinghair',
 'punching',
 'quarrel',
 'slapping',
 'stabbing',
 'strangle']

img = tform(Image.open(f_name)).to(device)
img.unsqueeze_(0)
_, pred = torch.max(model(img), 1)
print(classes[pred])
