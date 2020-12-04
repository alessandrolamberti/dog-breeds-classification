# Dog breeds classification with Transfer Learning

![alt text](https://www.repstatic.it/content/nazionale/img/2019/05/20/185502875-3c9b91ad-866a-44b6-9872-0fda39be6070.jpg)

## Problem statement

Given a simple, yet beautiful, dog picture, can a model guess his breed?

## Results

Managed to obtain a 99.8% accuracy training the model on the full dataset

## About the data

The original set comes from [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/overview). It consists of a collection of 10,000+ labelled images of 120 different dog breeds (that means we'll be dealing with unstructured data).

This kind of problem is called 'multi-class' image classification. It's multi-class because we're trying to classify multiple different breeds of dog.

## Workflow

* Prepare the data (preprocessing, the 3 sets, X & y)
* Choose and fit/train a model (TensorFlow Hub, tf.keras.applications, TensorBoard, EarlyStopping).
* Evaluating a model (making predictions, comparing them with the ground truth labels).
* Improve the model through experimentation (starting with a sample of images and slowly increase the number)
* Save our model

## Also had a bit of fun with custom predictions

![alt text](https://github.com/lamb-does-code/dog-breeds-classification/blob/main/images/custom_preds.png)
