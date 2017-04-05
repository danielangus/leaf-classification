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

In order to complete this problem, I will use the extracted features to build an effective leaf classifier. I chose not to use the images for two reasons. The first is that a typical effective strategy for classifying images is through a convolutional neural network (CNN). To implement a CNN, the images would have to be resized, which would eliminate some of the disinguishing characteristics of the leaves - their edges. The second reason for not using the image data is that the extracted features contain most or all of the information that can be gleaned from images, and so it would be redundant to make a classifier that uses them. 

An adequate solution to this problem is having at least 99% accuracy on a validation set and under 0.1 log loss on the testing set.

### Benchmark Model

As a sample solution to this problem, Kaggle provides a sample submission is which each test instance is assigned equal probability of being each of the 99 classes. Any submission scoring better than this can be said to be better than random, and so this is a good benchmark. The sample submission scores 4.6 on log loss.

### Evaluation Metrics

I will use log loss to grade the score on the testing set. Essentially, for each of the test instances, the classifier predicts the probability that the instance is one of the 99 classes. Log loss is then calculated as the negative sum of the average products between true classes and the log of predicted probabilities. This is effective for multi-class problems because, in practice, it serves to penalize incorrect class predictions [source: http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/]. 

### Project Design

The project will proceed as follows.

* Exploratory Analysis: examine the features and normalize to [0, 1]
* Step 1: train a variety of classifiers from sk-learn on the pre-extracted features. I will use stratified k-fold cross validation to ensure there are samples of each leaf among each fold. Grid search will be used lightly to make sure some important parameters are in the right neighbourhood. 
* Step 2: train a keras neural network on the features. It will be kept shallow and relatively simple to get a feel for how keras performs on this data set.
* Step 3: select the best-performing algorithm and perform a better optimization. If this is an sk-learn algorithm, this will involve providing several values for paramters in the grid search. If it is the keras neural network, it will involve adding more layers and adjusting parameters in order to better the algorithm's performance. 
