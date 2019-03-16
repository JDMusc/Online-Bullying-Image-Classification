The main script is test.py  
The following executes the script: 
```
>>> python test.py img_path
```

[Deep Residual Learning](https://arxiv.org/pdf/1512.03385.pdf) was our main guide for this project.  

"resnet-local" shows the development for our final model, but not the countless runs of trial and error.

We wrote a modified version of the [ResNet class that comes with PyTorch](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).  

Copyright Disclaimer: significant chunks of the code are based on [PyTorch's tutorials](https://pytorch.org/tutorials/).

There were several advantages to writing our own ResNet
* We addressed overfitting
    * We can specify a variable number of blocks (short-circuited layers)
    * We can specify the number of convolution channels in the beginning
    * We reduced the number of short-circuit blocks to 3, and the number of first-layer convolution channels to 32
    * We were acheiving poor validation set accuracy (45%) even when train set accuracy was close to 100%. 
    * We reduced number of layers and number of convolution channels in beginning which boosted validation accuracy 10%
* We plan to combine attention with ResNet as described in [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)

Our technology stack was PyTorch, with TensorBoard for visualization.