"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """

        number_of_samples = X.shape[0]
        D = np.ones(number_of_samples) / number_of_samples

        for t in range(self.T):
            h = self.WL(D, X, y)
            predictions = h.predict(X)
            differences_indicator = (predictions != y)
            epsilon = np.sum(D * differences_indicator)
            w = 0.5 * np.log((1/epsilon) - 1)
            D_unnormalized = D * np.exp(-w * y * predictions)
            D = D_unnormalized / np.sum(D_unnormalized)

            self.w[t] = w
            self.h[t] = h

        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predictions = np.array([self.h[t].predict(X) for t in range(max_t)])
        return np.sign(self.w[:max_t] @ predictions)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        adaboost_predictions = self.predict(X, max_t)
        return np.sum(adaboost_predictions != y) / y.size
