"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
import matplotlib.pyplot as plt
from ex4.ex4_tools import DecisionStump, decision_boundaries, generate_data, load_images
from ex4.adaboost import AdaBoost

import ex4.svm_perception_comperer as comperer

TRAIN_SET_SIZE = 5000
TEST_SET_SIZE = 200
T = 500
NOISE_RATIO = 0

X_train, y_train = generate_data(TRAIN_SET_SIZE, NOISE_RATIO)
X_test, y_test = generate_data(TEST_SET_SIZE, NOISE_RATIO)
adaboost_clf = AdaBoost(DecisionStump, T)
D = adaboost_clf.train(X_train, y_train)

T_TO_PLOT = np.array([5, 10, 50, 100, 200, 500])

def get_errors(X, y):
    errors = []
    for t in range(1,T+1):
        errors.append(adaboost_clf.error(X, y, t))
    return np.array(errors)


train_error = get_errors(X_train, y_train)
test_error = get_errors(X_test, y_test)

NOISES = [0.1, 0.4]


def Q4():
    comperer.q4()


def Q5():
    comperer.q5()


def Q8():
    all_t = np.arange(1, T+1)
    plt.plot(all_t, train_error, label="train set error")
    plt.plot(all_t, test_error, label="test set error")
    plt.title("AdaBoost Errors")
    plt.xlabel("adaBoost iterations")
    plt.ylabel("error rate")
    plt.legend()
    plt.show()


def Q9():
    for i in range(T_TO_PLOT.size):
        plt.subplot(2, 3, i+1)
        decision_boundaries(adaboost_clf, X_test, y_test, num_classifiers=T_TO_PLOT[i])
    plt.show()

def Q10():
    best_t = np.where(test_error == np.amin(test_error))[0][0] + 1

    decision_boundaries(adaboost_clf, X_train, y_train, num_classifiers=best_t)
    plt.show()
    decision_boundaries(adaboost_clf, X_test, y_test, num_classifiers=best_t)
    plt.show()


def Q11():
    normalized_D = D / np.max(D) * 10
    decision_boundaries(adaboost_clf, X_train, y_train, weights=normalized_D)
    plt.show()


def Q12():
    for n in NOISES:
        # globals not global
        X_train, y_train = generate_data(TRAIN_SET_SIZE, n)
        X_test, y_test = generate_data(TEST_SET_SIZE, n)
        adaboost_clf = AdaBoost(DecisionStump, T)
        D = adaboost_clf.train(X_train, y_train)

        def get_errors(X, y):
            errors = []
            for t in range(1,T+1):
                errors.append(adaboost_clf.error(X, y, t))
            return np.array(errors)


        train_error = get_errors(X_train, y_train)
        test_error = get_errors(X_test, y_test)


        # def Q8():
        all_t = np.arange(1, T + 1)
        plt.plot(all_t, train_error, label="train set error")
        plt.plot(all_t, test_error, label="test set error")
        plt.title("AdaBoost Errors with noise " + str(n))
        plt.xlabel("adaBoost iterations")
        plt.ylabel("error rate")
        plt.legend()
        plt.show()

        # def Q9():
        for i in range(T_TO_PLOT.size):
            plt.subplot(2, 3, i + 1)
            decision_boundaries(adaboost_clf, X_test, y_test, num_classifiers=T_TO_PLOT[i])
        plt.show()

        # def Q10():
        best_t = np.where(test_error == np.amin(test_error))[0][0] + 1

        decision_boundaries(adaboost_clf, X_train, y_train, num_classifiers=best_t)
        plt.show()
        decision_boundaries(adaboost_clf, X_test, y_test, num_classifiers=best_t)
        plt.show()

        # def Q11():
        normalized_D = D / np.max(D) * 10
        decision_boundaries(adaboost_clf, X_train, y_train, weights=normalized_D)
        plt.show()




def Q17():
    'TODO complete this function'


def Q18():
    'TODO complete this function'


if __name__ == '__main__':
    Q12()