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
    features = []
    targets = []
    classes = list(range(len(scales)))

    for i, (mean, std) in enumerate(zip(locs, scales)):
        features.extend(norm(mean, std).rvs(n))
        targets.extend([i]*n)

    return np.array(features), np.array(targets), np.array(classes)

def plot_data(
    features: np.ndarray,
    targets: np.ndarray,
    classes: np.ndarray
    ) -> np.ndarray:
    '''plots the data'''
    markers = ['o', 'x'] 
    y_values = [0]*len(features)
    for class_id in classes:
        class_features = [features[i] for i in range(len(features)) if targets[i] == class_id]
        class_y_values = [y_values[i] for i in range(len(y_values)) if targets[i] == class_id]
        
        plt.scatter(class_features, class_y_values, marker=markers[class_id], label=f'Class {class_id}')
    plt.show()
    



def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    data_points = features[targets == selected_class]
    return np.mean(data_points)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    data_points = features[targets==selected_class]
    return np.cov(data_points)


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

    likelihoods = norm.pdf(feature, class_mean, class_covar)
    return likelihoods
    
    

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
        mean = mean_of_class(train_features, train_targets, class_label)
        cov = covar_of_class(train_features, train_targets, class_label)
        means.append(mean)
        covs.append(cov)        

    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append(likelihood_of_class(test_features[i], means, covs))
 
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    prediction = np.argmax(likelihoods, 1)
    return prediction



if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """
    #np.random.seed(1234)
    #features, targets, classes = gen_data(50, [-1, 1], [np.sqrt(5), np.sqrt(5)])
    #(train_features, train_targets), (test_features, test_targets)\
    #= split_train_test(features, targets, train_ratio=0.8)
    #print(train_features, train_targets)
    #plot_data(features, targets, classes)
    #class_mean = mean_of_class(train_features, train_targets, 0)
    #class_cov = covar_of_class(train_features, train_targets, 0)
    #print(likelihood_of_class(test_features[0:3], class_mean, class_cov))
    #print(maximum_likelihood(train_features, train_targets, test_features, classes))
    #print(predict(maximum_likelihood(train_features, train_targets, test_features, classes)))

   # Section 8
    features, targets, classes = gen_data(50, [-4, 4], [np.sqrt(2), np.sqrt(2)])
    (train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.8)
    #
    all_likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    prediction = predict(all_likelihoods)
    #
    corr_pred = [0] * len(test_features.shape[0])
    no_pred = [0] * len(test_features.shape[0])
    #
 

    

    