#based off PyTorch implementation of ResNet with modifications
from toolz import pipe as p

from torch import nn


def makeConv2d(in_channels, out_channels, kernel_size=3, stride=1,
               padding = 1, bias = False):
    conv = nn.Conv2d(in_channels, out_channels, 
                     kernel_size = kernel_size, 
                     stride = stride,
                     padding = padding, bias = bias)
    
    nn.init.kaiming_normal_(conv.weight, mode='fan_out',
                            nonlinearity='relu')
    return conv


def makeBn2(num_channels):
    bn = nn.BatchNorm2d(num_channels)
    
    nn.init.constant_(bn.weight, 1)
    nn.init.constant_(bn.bias, 0)
    
    return bn


def preResLayer(out_channels = 64):
    return nn.Sequential(
        makeConv2d(3, out_channels, kernel_size=7, 
                   stride=2, padding=3),
        makeBn2(out_channels),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
    )

    
def postResLayer(in_channels, num_classes):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d( (1,1) ),
        Lambda(flatten),
        nn.Linear(in_channels, num_classes)
    )
    
    
#from PyTorch Website
class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

        
def flatten(x):
    return p(x,
      lambda _: _.size(0),
      lambda _: x.view(_, -1)
     )

    
class ResNet(nn.Module):
    def __init__(self, block_sizes, num_classes, in_channels = 64):
        super(ResNet, self).__init__()
        
        self.preres = preResLayer()
        
        blocks = []
        
        blocks.append(makeBlock(in_channels, in_channels, block_sizes[0], stride=1))
        
        for i in range(1, len(block_sizes)):
            out_channels = in_channels * 2
            blocks.append(makeBlock(in_channels, out_channels, block_sizes[i]))
            in_channels = out_channels
            
        self.blocks = nn.Sequential(*blocks)
        
        self.postres = postResLayer(out_channels, num_classes)
                
                
    def forward(self, x):
        return p(x,
                 self.preres,
                 self.blocks,
                 self.postres
                )


        
#unlike PyTorch, Block is defined as an array of layers
#ResNet paper defines layers as PyTorch defines blocks
def makeBlock(in_channels, out_channels, num_layers, stride=2):
    def makeLayer(i): 
        in_chan = in_channels if i == 0 else out_channels
        stri = stride if i == 0 else 1
        
        return ResLayer(in_chan, out_channels, stride=stri)
    
    return nn.Sequential(*[makeLayer(i) for i in range(0, num_layers)])


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResLayer, self).__init__()
        
        self.conv1 = makeConv2d(in_channels, out_channels,
                                  stride = stride)
        self.bn1 = makeBn2(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = makeConv2d(out_channels, out_channels)
        self.bn2 = makeBn2(out_channels)
        
        self.resizeInput = self.resizeInputGen(in_channels, out_channels, stride)
        self.stride = stride
     
    
    def resizeInputGen(self, in_channels, out_channels, stride):
        resizeInput = lambda _: _
        
        if in_channels != out_channels or stride != 1:
            resizeInput = nn.Sequential(
                makeConv2d(
                    in_channels, out_channels, kernel_size = 1, stride = stride, padding=0),
                makeBn2(out_channels)
            )
            
        return resizeInput
    
    
    def forward(self, x):
        def addInput(processed_x): 
            return processed_x + self.resizeInput(x)
        
        return p(x,
                 self.conv1,
                 self.bn1,
                 self.relu,
                 self.conv2,
                 self.bn2,
                 addInput,
                 self.relu
                )
        