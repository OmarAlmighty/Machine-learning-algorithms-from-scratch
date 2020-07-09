# **Project description**
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
