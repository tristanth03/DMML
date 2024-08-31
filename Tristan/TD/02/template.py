# Author: 
# Date:
# Project: 
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


if __name__ == "__main__":
    """
    Keep all your test code here or in another file.
    """

    
   


    
    # features,targets,classes = gen_data(50, [-1,2], [np.sqrt(5),1])
    # (train_features, train_targets), (test_features, test_targets)\
    #     = split_train_test(features, targets, train_ratio=0.8)
    # class_mean = mean_of_class(train_features, train_targets, 0)
    # class_cov = covar_of_class(train_features, train_targets, 0)
    # l = maximum_likelihood(train_features,train_targets,test_features,classes)
    # print(l)
    # print(predict(l))
    
    # for _class in range(features.shape[1]):

    #     # Some general shapes for markers, one for even nums and one for odd
    #     if _class % 2 == 0:
    #         marker = 'o'
    #     else:
    #         marker = 'x'
        
    #     plt.scatter(features[:,_class],np.array([0]*features.shape[0]),marker=marker,label=fr"Class {_class}")
    # plt.legend()
    # plt.title("Test and train data, seed: 1234")
    # plt.xlabel("Numerical value")
    # plt.show()


    # mus = [-4,4]
    # var_s = [np.sqrt(2),np.sqrt(2)]
    # accuracy = []
    # for i in range(len(mus)):
    #     features,targets,classes = gen_data(50, [mus[i]], [var_s[i]])
    #     (train_features, train_targets), (test_features, test_targets)\
    #     = split_train_test(features, targets, train_ratio=0.8)
    #     max_like = maximum_likelihood(train_features,train_targets,test_features,classes)
    #     class_pred = predict(max_like)
    #     correct_pred = 0
    #     incorrect_pred = 0
    #     for p in range(len(test_targets)):
    #         if test_targets[p] == class_pred[p]:
    #             correct_pred += 1
    #         else:
    #             incorrect_pred += 1
    #     accuracy.append(correct_pred/len(test_targets))
    
    # print(accuracy)
    


    # Define the parameters for the two classes
    mus = [-4, 4]
    scales = [np.sqrt(2), np.sqrt(2)]

    # Generate data from both distributions, treating each as a separate class
    features, targets, classes = gen_data(50, mus, scales)

    # Split the dataset into training and testing sets
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.8)

    # Estimate likelihood and predict the class for each test feature
    max_like = maximum_likelihood(train_features, train_targets, test_features, classes)
    class_pred = predict(max_like)

    # Calculate the accuracy by comparing predicted labels with actual labels
    correct_pred = np.sum(test_targets == class_pred)
    accuracy = correct_pred / len(test_targets)

    print(f"Accuracy: {accuracy}")




    