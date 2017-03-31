# Machine Learning Engineer Nanodegree

## Capstone Proposal

Daniel Beasley
April 2017

For the Udacity Machine Learning Engineer Nanodegree Capstone Project I will complete the Kaggle Leaf Classification challenge. 

### Domain Background

The goal of this project is to classify leaves from various species of plants. This is a difficult task for humans to perform since there are estimated to be nearly half a million species of plants in the world (per the challenge description). With machine learning-based automation of the classification process, we can gain insight into the spread of plant species and population counts.

### Problem Statement

The problem is to correctly identify 99 different species of plants from their leaves.

### Datasets and Inputs

There are two datasets with which to apply classifiers. The first is 1,584 images of leaves - 16 images for each of the 99 species). The second is a csv file of extracted features. 
There are three sets of features in the csv file:
* shape - describes the shape of the leaf
* margin - describes the fine details of the leaf
* texture - descibes the interior structure of the leaf

Each of the three feature sets contains 64 columns, for a total of 192 extracted features.

### Solution Statement

An adequate solution to this problem is having at least 99% accuracy on a validation set and under 0.05 log loss on the testing set. 

### Benchmark Model

As a sample solution to this problem, Kaggle provides a sample submission is which each test instance is assigned equal probability of being each of the 99 classes. Any submission scoring better than this can be said to be better than random, and so this is a good benchmark. The sample submission scores 4.6 on log loss.

### Evaluation Metrics

