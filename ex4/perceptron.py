import numpy as np
from theano.gof.tests.test_destroyhandler import add_in_place


class perceptron:

    def __init__(self):
        self.w = None


    def fit(self, X, y):
        number_of_features = X.shape[1]
        self.w = np.zeros(number_of_features + 1)
        NO_SAMPLE_NEED_FIX = -1

        X = self.add_intercept_to_samples(X)

        while True:
            i = self.find_unsatisfying_sample(X, y)

            if i == NO_SAMPLE_NEED_FIX:
                return self.w

            self.w = self.w + y[i] * X[i, :]


    def add_intercept_to_samples(self, X):
        number_of_samples = X.shape[0]
        free_variables = np.ones(number_of_samples)
        X = np.hstack((X, free_variables))
        return X


    def find_unsatisfying_sample(self, X, y):
        number_of_samples = X.shape[0]

        for i in range(number_of_samples):
            if y[i] * np.inner(self.w, X[i, :] <= 0):
                return i

        return -1


    def predict(self, x):
        x = np.hstack((x,1))
        return np.sign(np.inner(w, x))


    def predict_many(self, X):
        X = self.add_intercept_to_samples(X)
        return np.sign(X @ self.__w)


    def get_weights_and_intercept(self):
        return self.__w[:-1], self.__w[-1]