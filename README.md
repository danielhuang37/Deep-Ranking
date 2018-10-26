# Deep-Ranking

### Training

main.py : trains the model to learn the tripleloss

### Inference

1.	Load the best checkpoint from training and use the model as predictor
2.  Compute the embedding of each testing output, and then save them into files
3.  Loads the train embedding and fit it into KNN.
4.  Loads the test embedding and find the top 30 closest train embedding
5.  Compute top-30 precision by suming the correct labels in the predicted labels  


### Demo

For 5 of the validation images, find the top 10 images closest to it (including its distance) and bot 10 images furtherest from it according to the model.
=======
This repo is my attempt to reproduce "Learning Fine-grained Image Similarity with Deep Ranking" (https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf) in Pytorch.

## Task 
Given an image, find images that are similar to it in terms of Eculidean distance. The accuracy is calcuated using top-30 precision.

## Dataset
[Tiny-Imagenet](href=https://tiny-imagenet.herokuapp.com/): A smaller scale ImageNet dataset consists of 200 classes and each image has size 64 x 64 x 3.

## Methods and results
### Training
I used pre-trained Resnet-101 and trained for 15 epochs with 0.01 learning rate. 


## Visualization
### Good case
![Orginal Image](https://github.com/danielhuang37/Deep-Ranking/blob/master/img/n04399382-orginal.png)
![Closest Images](https://github.com/danielhuang37/Deep-Ranking/blob/master/img/n04399382-top.png)

### Bad Case
![Orginal Image](https://github.com/danielhuang37/Deep-Ranking/blob/master/img/n04070727-orginal.png)
![Closest Images](https://github.com/danielhuang37/Deep-Ranking/blob/master/img/n04070727-top.png)

## Improvement
#### Use multi-scale network architecture:
As mentioned in the paper, we could
use multi-scale network architecture instead of single scale ConvNet because
single scale ConvNet has very strong invariance encoded in the model. In the
task of ranking image similarity, having strong invariance in the model can be
harmful for fine-grained image similarity tasks.
#### Include in-class negative samples:  
Since images in the same class can have
high variance, the model should rank the in-class images that are more similar
than those that are not so much.

