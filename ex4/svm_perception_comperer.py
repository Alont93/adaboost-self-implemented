import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from ex4 import perceptron

SVM_C = 10 * (10**10)
DATA_MEAN = np.zeros(2)
DATA_COV = np.eye(2)

DATA_SEPERTOR = [0.3, -0.5]
DATA_SEPERTOR_FREE_VAR = 0.1

SAMPLE_SIZES = [5, 10, 15, 25, 70]

PLOT_RANGE = 3
PLOT_DENSITY = 1000

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


q4()