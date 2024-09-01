# Author: Tristan Thordarson
# Date: 31.08.2024
# Project: 02
# Acknowledgements: 
#
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

def gen_data(
    n: int,
    locs: np.ndarray,
    scales: np.ndarray
) -> np.ndarray:
    '''
    Return n data points, their classes and a unique list of all classes, from each normal distributions
    shifted and scaled by the values in locs and scales
    '''
    data = []
    labels = []
    classes = []
    for i in range(len(locs)):
        classes.append(i)
        labels.extend([i]*n)
        data.extend(norm.rvs(locs[i],scales[i],size=n)) 


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
    selected_features = features[targets==selected_class] 
    mu = np.mean(selected_features,0)
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
    selected_features = features[targets==selected_class]
    cov = np.cov(selected_features,rowvar=False)
    return cov


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> np.ndarray:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    p = norm.pdf(feature,class_mean,class_covar) # assuming we have a 1-dimensional input (i.e. we iterate)
    p_multi = np.prod(p)
    return p_multi


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
        means.append(mean_of_class(train_features,train_targets,class_label))
        covs.append(covar_of_class(train_features,train_targets,class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        sample_likelihoods = []
        for class_label in classes:
            sample_likelihoods.append(likelihood_of_class(test_features[i],means[class_label],covs[class_label]))
        likelihoods.append(sample_likelihoods)
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    pred = np.argmax(likelihoods,1)
    return pred



def plot_data():
    features, targets, classes = gen_data(50, [-1, 1], [np.sqrt(5), np.sqrt(5)])
    for _class in classes:
        if _class % 2 == 0: # even -> 0 odd -> x
            marker = 'o'
        else:
            marker = 'x'
        class_features = features[targets == _class]

        plt.scatter(class_features, np.zeros_like(class_features), marker=marker, label=fr"Class {_class}")
    
    plt.legend()
    plt.title("Test and train data, seed: 1234")
    plt.xlabel("Numerical value")
    plt.show()


def case_study():
    mus = [-4,4]
    var_s = [np.sqrt(2),np.sqrt(2)]
    features,targets,classes = gen_data(50, mus, var_s)
    (train_features, train_targets), (test_features, test_targets)\
        = split_train_test(features, targets, train_ratio=0.8)
    maxi_liki = maximum_likelihood(train_features,train_targets,test_features,classes)
    class_hat = predict(maxi_liki)
    
    class_count = [0]*(classes[-1]+1)
    correct_preds = [0]*(classes[-1]+1)
    accuracy = [0]*(classes[-1]+1)
    for i in range(len(test_targets)):
        _class = test_targets[i]
        class_count[_class] += 1
        if _class == class_hat[i]:
            correct_preds[_class] += 1
        accuracy[_class] = correct_preds[_class] / class_count[_class]
    
    print(accuracy)

if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """

    np.random.seed(1234)
   


    
    # features,targets,classes = gen_data(50, [-0,0], [np.sqrt(2),np.sqrt(2)])
    # (train_features, train_targets), (test_features, test_targets)\
    #     = split_train_test(features, targets, train_ratio=0.8)
    # class_mean = mean_of_class(train_features, train_targets, 0)
    # class_cov = covar_of_class(train_features, train_targets, 0)
    # l = maximum_likelihood(train_features,train_targets,test_features,classes)
    # print(l)
    # print(predict(l))

    case_study()