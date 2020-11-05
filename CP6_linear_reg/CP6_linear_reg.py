"""
Implementation of linear regression from scratch 
test code on USA housing dataset
Install kaggle and download data first
"""
# import os
# os.system(
#     "kaggle d download aariyan101/usa-housingcsv -p ./CP6_linear_reg --unzip"
# )

import csv
import random
import math
import operator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Data:
    def __init__(self, path, drop=None):
        self.data = pd.read_csv(path)
        self.length = len(self.data)
        self.n_features = len(self.data.columns) - 1
        if drop:
            self.drop(columns=drop)

    def drop(self, columns):
        if isinstance(columns, str):
            self.data.drop(columns=[columns], inplace=True)
        else:
            self.data.drop(columns=columns, inplace=True)
        return self

    @staticmethod
    def normalize(X):
        X -= X.mean(axis=0, keepdims=True)
        X /= np.linalg.norm(X, axis=0, keepdims=True)
        return X

    @staticmethod
    def standardize(X):
        X -= X.mean(axis=0, keepdims=True)
        X /= X.std(axis=0, keepdims=True)
        return X

    def train_test_split(self, label, test_size=0.2, seed=None):
        if seed:
            np.random.seed(seed=seed)
        self.features, self.labels = self.data.drop(columns=[label]), self.data[label]
        test_idx = np.random.choice(
            self.features.index, size=int(test_size * self.length), replace=False
        )
        X_train, X_test = self.features.drop(test_idx), self.features.loc[test_idx]
        y_train, y_test = self.labels.drop(test_idx), self.labels.loc[test_idx]
        return X_train, X_test, y_train, y_test

    def pair_plot(self):
        sns.pairplot(self.data)

    def dist_plot(self, name):
        sns.distplot(self.data[name])


class LinearRegression:
    def __init__(
        self, seed=None, fit_intercept=True, normalize=False, standardize=False
    ):
        self.params = {
            "seed": seed,
            "fit_intercept": fit_intercept,
            "normalize": normalize,  # if both normalize and standardize, then normalize (see fit and predict)
            "standardize": standardize,
        }
        self._coef = None

    def get_params(self):
        return self.params

    def set_params(self, **params):
        for key, val in params:
            if key in self.params.keys():
                self.params[key] = val
            else:
                raise KeyError("This param does not exist: {}".format(key))

    @property
    def coef(self):
        return self._coef

    @staticmethod
    def rank(X):
        return np.linalg.matrix_rank(X)

    def fit(self, X, y):
        """
        Fit ordinary least squares to X, y with usual assumptions:
        1. data determined without error
        2. E[error_i] = 0
        3. Var(error_i) = sigma**2
        4. cov(error_i, error_j) = 0
        5. cov(X_i, error_j) = 0
        6. error ~ N(0, sigma**2 I_n)
        7. no colinearity among variables: X has full rank p
        8. n > p + 1
        """
        X = X.to_numpy(dtype=float)
        if self.params["normalize"]:
            self.scaler_means = X.mean(axis=0)
            self.scaler_divs = np.linalg.norm(X, axis=0)
            X -= self.scaler_means
            X /= self.scaler_divs
        elif self.params["standardize"]:
            self.scaler_means = X.mean(axis=0)
            self.scaler_divs = X.std(axis=0)
            X -= self.scaler_means
            X /= self.scaler_divs
        if self.params["fit_intercept"]:
            length, _ = X.shape
            intercept = np.ones((length, 1))
            X = np.concatenate((intercept, X), axis=1)
        y = y.to_numpy(dtype=float)
        self._coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        return self

    def predict(self, X):
        """ Predict with norm and starndard if necessary """
        if self.params["normalize"] or self.params["standardize"]:
            X -= self.scaler_means
            X /= self.scaler_divs
        if self.params["fit_intercept"]:
            length, _ = X.shape
            intercept = np.ones((length, 1))
            X = np.concatenate((intercept, X), axis=1)
        return np.dot(X, self._coef)

    def score(self, X, y, metric="r2"):
        y_true = y
        y_pred = self.predict(X)
        if metric == "r2":
            score = self.r2_score(y_true, y_pred)
        elif metric == "mse":
            score = self.mse_score(y_true, y_pred)
        elif metric == "rmse":
            score = math.sqrt(self.mse_score(y_true, y_pred))
        return score

    @staticmethod
    def r2_score(y_true, y_pred):
        """ compute R2 score """
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1 - u / v

    @staticmethod
    def mse_score(y_true, y_pred):
        """ COmpute mse """
        return ((y_true - y_pred) ** 2).mean()


if __name__ == "__main__":
    d = Data("./CP6_linear_reg/USA_housing.csv", drop="Address")
    X_train, X_test, y_train, y_test = d.train_test_split(
        label="Price", test_size=0.2, seed=100
    )
    lr = LinearRegression(normalize=True).fit(X_train, y_train)
    r2 = lr.score(X_test, y_test)
    print("{:.2f}".format(r2))
