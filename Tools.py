import math
from math import sqrt

import numpy as np
import random


# Separate features from classes
def get_features(dataset, ind_lst, ind_class):
    x = dataset.iloc[:, ind_lst].values
    y = dataset.iloc[:, ind_class].values
    x = x.astype(float)
    return x, y


# Split into train set and test set
def train_test_split(dataset, test_size):
    train_set = dataset.values.tolist()
    test_set = []
    split = test_size * len(train_set)
    while len(test_set) < split:
        test_set.append(train_set.pop(random.randrange(len(train_set))))
    return train_set, test_set


# Convert string column to float
def str_to_float(dataset, column):
    for instance in dataset:
        instance[column] = float(instance[column].strip())


# Convert string column to integer
def str_to_int(dataset, column):
    lbls = [instance[column] for instance in dataset]
    unique = set(lbls)
    result = dict()
    for i, val in enumerate(unique):
        result[val] = i
    for instance in dataset:
        instance[column] = result[instance[column]]
    return result


# Find the min and max values for each column
def get_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [instance[i] for instance in dataset]
        minimum = min(col_values)
        maximum = max(col_values)
        minmax.append([minimum, maximum])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset):
    minmax = get_minmax(dataset)
    for instance in dataset:
        for i in range(len(instance)):
            instance[i] = (instance[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Calculate the accuracy
def get_accuracy(y_true, y_predicted):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_predicted[i]:
            correct += 1
    return correct / len(y_true) * 100.0


# Calculate the mean of a list of numbers
def mean(col):
    return sum(col) / float(len(col))


# Calculate the standard deviation of a list of numbers
def stdev(col):
    avg = mean(col)
    variance = sum([(x - avg) ** 2 for x in col]) / float(len(col) - 1)
    return sqrt(variance)


# Calculate the Gaussian probability distribution function for x
def estimate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent