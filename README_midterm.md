The main script is test.py  
The following executes the script: 
```
>>> python test2.py img_path
```

[Deep Residual Learning](https://arxiv.org/pdf/1512.03385.pdf) was our main guide for this project.  

"resnet-local" shows the development for our final model, but not the countless runs of trial and error.  

We wrote a modified version of the [ResNet class that comes with PyTorch](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). The Residual Net code file is _localResnet.py_.  

Copyright Disclaimer: significant chunks of the code are based on [PyTorch's tutorials](https://pytorch.org/tutorials/).  

There were several advantages to writing our own ResNet
* addressing overfitting
    * Added drop out support 
        * we settle with .2
    * The number of convolution channels in the beginning can be specified so model has less parameters
        * We reduced number of first-layer convolution channels to 32
    * Will be easier to combine attention with ResNet as described in [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)  
        

Performance results  
    * 92% accuracy training set, 64% accuracy validation set  


 Our technology stack was PyTorch, with TensorBoard for visualization.
