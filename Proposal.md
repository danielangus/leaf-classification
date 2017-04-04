# Machine Learning Engineer Nanodegree

## Capstone Proposal

Daniel Beasley
April 2017

For the Udacity Machine Learning Engineer Nanodegree Capstone Project I will complete the Kaggle Leaf Classification challenge. 

### Domain Background

The goal of this project is to classify leaves from various species of plants. This is a difficult task for humans to perform since there are estimated to be nearly half a million species of plants in the world (per the challenge description). With machine learning-based automation of the classification process, we can gain insight into the spread of plant species and population counts.

### Problem Statement

The problem is to correctly classify 99 different species of plants from their leaves. Each leaf is scanned, white on a black background. There is also a set of 192 extracted features from each leaf to aid in classification. This is a multi-class problem with 99 output classes. 

### Datasets and Inputs

There are two datasets with which to apply classifiers. The first is 1,584 images of leaves - 16 images for each of the 99 species). The second is a csv file of extracted features. 
There are three sets of features in the csv file:
* shape - describes the shape of the leaf
* margin - describes the fine details of the leaf
* texture - descibes the interior structure of the leaf

Each of the three feature sets contains 64 columns, for a total of 192 extracted features. The data can be found here: https://www.kaggle.com/c/leaf-classification/data

### Solution Statement

In order to complete this problem, I will use both the extracted features and images to create an effective classifier. An adequate solution to this problem is having at least 99% accuracy on a validation set and under 0.1 log loss on the testing set.

### Benchmark Model

As a sample solution to this problem, Kaggle provides a sample submission is which each test instance is assigned equal probability of being each of the 99 classes. Any submission scoring better than this can be said to be better than random, and so this is a good benchmark. The sample submission scores 4.6 on log loss.

### Evaluation Metrics

Kaggle uses log loss to grade the score on the testing set. Essentially, for each of the test instances, the classifier predicts the probability that the instance is one of the 99 classes. Log loss is then calculated as the negative sum of the average products between true classes and the log of predicted probabilities. In practice, this serves to penalize incorrect class predictions [source: http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/]

### Project Design

The project will proceed as follows.

* Exploratory Analysis: examine the features and normalize to [0, 1]
* Step 1: train a variety of classifiers from sk-learn on the pre-extracted features. I will use stratified k-fold cross validation to ensure there are samples of each leaf among each fold. Grid search will be used lightly to make sure some important parameters are in the right neighbourhood. 
* Step 2: select the best-performing algorithm and perform a better optimization. This will involve providing several values for paramters in the grid search. I will also try scorers other than accuracy, like F-beta and log loss.
* Step 3: apply a TensorFlow neural network to classify the images. First I will try a deep neural network on the pre-extracted features, and then perhaps a convolutional neural network on the images. 
