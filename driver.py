# -*- coding: utf-8 -*-
"""Driver.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iImVJGARPncwOu6dgUCNrTvH7PPwqmxc

# **ML practice project description**
This project is designed to train any dataset on the following classifiers: **KNN**, **Naive Bayes**, and **Decision Tree**

The project is organized as follows:


1.   **KNN.py** class, to classify the dataset using k-nearset neighbor classifier.
2.   **NaibeBayes.py** class, to classify the dataset using naive bayes classifier.
3.   **DecisonTree.py** class, to classify the dataset using decision tree classifier.
4.   **Tools.py** class, this class contains general functions used by each classifier such as: 

    *   *train_test_split(dataset, test_size)*: to split dataset to train set and test set given split size
    *   *get_accuracy(y_true, y_predicted)*: to calculate accuracy on a test set by counting the number of correctly classified objects then divide by the total number of objects in the testset.
    *   *euclidean_distance(row1, row2)*: to estimate the eculdian distance between two instances. Used in KNN classifier. 
    *   *estimate_probability(x, mean, stdev)*: to estimate gaussian probability distruiton of an object x given mean and standard deviation.

5.    **Plot.py** class, to visualize dataset.
6.    **Driver.ipynb** notebook, to visualize, trian, and test classifiers on the dataset.

**The motivations behind using KNN is:**
1. Easy to implement
2. Makes no assumption about the underlying data pattern which means that it's non-parametric.
3. Can be used for both classification and Regression problems.
4. Based on feature similarity.
5. There is no explicit training phase or it is very minimal.

**The motivations behind using Naive Bayes is:**
1. Assumes that the features are independent.
2. Simple and fast.
3. Can be used for both binary and mult-iclass classification problems.
4. Can make probabilistic predictions.
5. Not sensitive to irrelevant features.

**The motivations behind using Naive Bayes is:**
1. Can handle continuous-valued, discrete, and categorical; there is no need to convert one type into another.
2. An intuitve classifier, as it can be traced as a sequence of choices.
3. The operation is very fast.
4. does not require normalization of data. 
5. Unstable classifier.

### Import modules
"""

import pandas as pd
from Plot import Plot
import Tools
from KNN import KNN
from NaiveBayes import NaiveBayes
from DecisonTree import DecisonTree
print('Done importing')

"""### Load data
 load dataset
 
 names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'class']
 
 dataset = pd.read_csv(filename.csv, names=[feature names])
"""

names = []
num_of_features = 13
for i in range(num_of_features):
    x = 'x'+str(i+1)
    names.append(x)
names.append('class')
dataset = pd.read_csv('dataset.csv', names=names)
# IF DATASET FILE IS NOT SEPARATED BY COMMAS
# dataset =pd.read_csv('output_list.txt', sep=" ", names=names)
print(dataset.head())

"""### Extract labels from dataset"""

lbls = dataset['class'].unique()
print(lbls)

"""### visualize the whole dataset"""

vis = Plot(lbls)
vis.visulaize(dataset,'x1','x5')

"""### Shuffle dataset
frac=1 means return all rows (in random order).
"""

dataset_shuffle = dataset.sample(frac=1).reset_index(drop=True)
print(dataset_shuffle.head())

"""### Create Plot object, and plot sample data after shuffling
plt.show_plt(feature1_for_x-axis, feature2_for_y-axis, number_of_samples, shuffled_dataset)
"""

plt = Plot(lbls)
plt.show_plot('x1', 'x5', 20, dataset_shuffle)



"""### separate features from class label"""

#x, y = Tools.get_features(dataset_shuffle,[0,1,2,3] ,4)
#print(x[0:5][:])
#print(y[0:5][:])

"""### train/test split"""

split_size = 0.40
train_set,test_set = Tools.train_test_split(dataset_shuffle,split_size)
print(len(train_set))
print(len(test_set))

"""### If there is non-mumerical labels, cast class labels to integers"""

# Tools.str_to_int(train_set,len(train_set[0])-1)
# Tools.str_to_int(test_set,len(test_set[0])-1)

"""### Run classifier"""

knn_classifier = KNN(train_set,test_set,3)
accuracy_knn = knn_classifier.run()
print('KNN: ',accuracy_knn)

dt_classifier = DecisonTree(train_set,test_set,5,10)
accuracy_dt = dt_classifier.run()
print('Decision Tree: ',accuracy_dt)

nb_classifier = NaiveBayes(train_set,test_set)
accuracy_nb = nb_classifier.run()
print('Naive Bayes: ',accuracy_nb)