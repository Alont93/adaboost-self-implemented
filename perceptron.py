import numpy as np

class perceptron:

    def __init__(self):
        self.__w = None


    def fit(self, X, y):
        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        self.__w = np.zeros(number_of_features + 1)

        X = self.add_intercept_to_samples(X)

        while True:
            still_converging = False
            for i in range(number_of_samples):
                if y[i] * np.inner(self.__w, X[i, :]) <= 0:
                    self.__w = self.__w + y[i] * X[i, :]
                    still_converging = True

            if not still_converging:
                break


    def add_intercept_to_samples(self, X):
        number_of_samples = X.shape[0]
        free_variables = np.ones(number_of_samples).reshape(number_of_samples, 1)
        X = np.hstack((X, free_variables))
        return X


    def predict(self, x):
        x = np.hstack((x,1))
        return np.sign(np.inner(self.__w, x))


    def predict_many(self, X):
        X = self.add_intercept_to_samples(X)
        return np.sign(X @ self.__w)


    def get_weights(self):
        return self.__w[:-1]


    def get_intercept(self):
        return self.__w[-1]