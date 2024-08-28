# Author: 
# Date:
# Project: 
# Acknowledgements: 
#
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    data = norm.rvs(locs,scales,size=(n,len(locs)))
    labels = []
    classes = []
    for i in range(len(locs)):
        classes.append(i)
        for j in range(len(data)):
            labels.append(i)


    return np.array(data),np.array(labels),np.array(classes)


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    
    selected_nums = len(targets)
    mu = np.mean(features[selected_nums,selected_class],0)
    return mu





def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    ...


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    ...


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        ...
    likelihoods = []
    for i in range(test_features.shape[0]):
        ...
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    ...


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """


    

    features,targets,classes = gen_data(50, [-1, 1], [np.sqrt(5), np.sqrt(5)])
    (train_features, train_targets), (test_features, test_targets)\
        = split_train_test(features, targets, train_ratio=0.8)
    print(mean_of_class(train_features,test_features,0))
    # for _class in range(features.shape[1]):

    #     # Some general shapes for markers, one for even nums and one for odd
    #     if _class % 2 == 0:
    #         marker = 'o'
    #     else:
    #         marker = 'x'
        
    #     plt.scatter(features[:,_class],np.array([0]*features.shape[0]),marker=marker,label=fr"Class {_class}")
    # plt.legend()
    # plt.show()





    