The main script for the final is testfinal.py  
The following executes the script: 

The model can be downloaded from here and should be placed in the same directory as `testfinal.py`:
https://drive.google.com/open?id=1-RWh4M3hgomeXbdi9C94k4iAKMZntDJ6

```
>>> python testfinal.py img_path
```

We tried 2 approaches. 
* Approach 1 was to perform transfer learning by starting at the deepest layers and steadily allow updating of less deep layers.
* Approach 2 was to allow all the layers to update

Approach 2 had better results as discussed in our paper.

Performance results for 10 class prediction 
* 97.2% accuracy training set
* 77.6% accuracy validation set  
* 78.2% accuracy 55 sample left-out test set


Performance results for non-bully (1) vs bully (0) 
* 96.2% TPR/99.7% Accuracy, training set, non-bullying defined as positive
* 57.1% TPR/97.6% Accuracy, validation set  
* 100% TPR/98.2% Accuracy, 55 sample left-out test set


__Important Reference Papers__  

[VGG](https://arxiv.org/abs/1409.1556)  
[How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)
