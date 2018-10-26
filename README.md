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

