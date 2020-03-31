import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import perceptron

SVM_C = 10 * (10**10)
DATA_MEAN = np.zeros(2)
DATA_COV = np.eye(2)

DATA_SEPERTOR = [0.3, -0.5]
DATA_SEPERTOR_FREE_VAR = 0.1

SAMPLE_SIZES = [5, 10, 15, 25, 70]

PLOT_RANGE = 3
PLOT_DENSITY = 1000

K = 10000
NUMBER_OF_TRYS = 500

def create_svm(X, y):
    clf = SVC(C=SVM_C, kernel='linear')
    clf.fit(X, y)
    return clf


def create_perceptron(X, y):
    clf = perceptron.perceptron()
    clf.fit(X, y)

    return clf


def get_data(m):
    X = np.random.multivariate_normal(DATA_MEAN, DATA_COV, m)
    y = determine_label(X)
    # labeler_func = np.vectorize(determine_label)
    # y = labeler_func(X)

    return X, y


def determine_label(x):
    return np.sign(np.inner(DATA_SEPERTOR, x) + DATA_SEPERTOR_FREE_VAR)


def draw_hyperplane(w, b, plot_label):
    points = np.linspace(PLOT_RANGE * -1, PLOT_RANGE, 1000)
    a = - w[0] / w[1]
    b = b / w[1]
    values = a * points - b

    plt.plot(points, values, label=plot_label)



def q4():
    for m in SAMPLE_SIZES:

        X, y = get_data(m)

        plt.scatter(X[:,0], X[:,1], c=y)

        draw_hyperplane(DATA_SEPERTOR, DATA_SEPERTOR_FREE_VAR, "True Hypothesis")

        svm_clf = create_svm(X, y)
        draw_hyperplane(svm_clf.coef_[0], svm_clf.intercept_[0], "SVM Hypothesis")

        perceptron_clf = create_perceptron(X, y)
        draw_hyperplane(perceptron_clf.get_weights(), perceptron_clf.get_intercept(), "Perceptron Hypothesis")

        plt.title('Hyperlines for hypothesises on the data. m='+str(m))
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.show()



def q5():

    svm_results = np.zeros((NUMBER_OF_TRYS, len(SAMPLE_SIZES)))
    perceptron_results = np.zeros((NUMBER_OF_TRYS, len(SAMPLE_SIZES)))

    for i in range(NUMBER_OF_TRYS):

        m_number = 0
        for m in SAMPLE_SIZES:

            X_train, y_train = draw_data_with_two_classes(m)
            X_test, y_test = draw_data_with_two_classes(K)

            svm_clf = create_svm(X_train, y_train)
            svm_prediction = svm_clf.predict(X_test)
            right_svm_percentage = np.count_nonzero(np.abs(svm_prediction - y_test) == 0) / K
            svm_results[i, m_number] = right_svm_percentage

            perceptron_clf = create_perceptron(X_train, y_train)
            perceptron_prediction = perceptron_clf.predict_many(X_test)
            right_perceptron_percentage = np.count_nonzero(np.abs(perceptron_prediction - y_test) == 0) /  K
            perceptron_results[i, m_number] = right_perceptron_percentage

            m_number += 1

    sam_mean_accuracies = np.mean(svm_results, axis=0)
    perception_mean_accuracies = np.mean(perceptron_results, axis=0)

    plt.plot(SAMPLE_SIZES, sam_mean_accuracies, label="svm")
    plt.plot(SAMPLE_SIZES, perception_mean_accuracies, label="perceptron")
    plt.title('SVM and Perceptron accuracy as function of training data size')
    plt.xlabel("training data size")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def draw_data_with_two_classes(m):
    no_two_classes_exists = True
    while no_two_classes_exists:
        X, y = get_data(m)
        if np.unique(y).size == 2:
            no_two_classes_exists = False

    return X, y
