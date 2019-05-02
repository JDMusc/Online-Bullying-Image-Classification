from PIL import Image
import sys

import torch

import preprocessing as pp
import vggTransfer

f_name = str(sys.argv[1])

device = torch.device("cuda")

vgg = vggTransfer.loadVgg(n_classes = 10)
if torch.cuda.is_available():
    vgg = vgg.to(device)

vgg.load_state_dict(torch.load('model_11.pt'))
vgg.eval()

tform = pp.createDataTransforms()['val']
img = tform(Image.open(f_name))
if torch.cuda.is_available():
    img = img.to(device)

img.unsqueeze_(0)

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

_, pred = torch.max(vgg(img), 1)
print(classes[pred])