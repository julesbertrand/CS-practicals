import pandas as pd
import random
import math
import operator


class Data:
    def __init__(self):
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv("./CP4_KNN/iris.csv")
        self.data.columns = ["seplen", "sepwid", "petlen", "petwid", "label"]

    def train_test_split(self, test_size, random_state=None):
        if random_state:
            random.seed(random_state)
        n = self.data.shape[0]
        test_idx = random.sample(list(self.data.index), int(test_size * n))
        X, y = self.data.drop(columns=["label"]), self.data["label"]
        X_train, X_test = X.drop(index=test_idx), X.iloc[test_idx]
        y_train, y_test = y.drop(index=test_idx), y.iloc[test_idx]
        return X_train, X_test, y_train, y_test


class KNN:
    def __init__(self, n_neighbors=5):
        self.params = {"n_neighbors": n_neighbors}

    @property
    def n_neighbors(self):
        return self.params["n_neighbors"]

    def get_params(self):
        return self.params

    def set_params(self, params):
        if not isinstance(params, dict):
            raise TypeError("Expected: dict")
        if not params:
            return self
        for key, val in params.items():
            if key in self.params.keys():
                self.params[key] = val
            else:
                raise KeyError("This parameter does not exist: {s}".format(key))

    def fit_scaler(self, data):
        self.mins = data.min()
        self.maxs = data.max()

    def scale(self, data):
        data = data.sub(self.mins, axis=1).div(self.maxs, axis=1)
        return data

    def fit(self, X, y):
        self.fit_scaler(X)
        self.features = self.scale(X)
        self.labels = y

    def predict(self, new_X):
        if not isinstance(new_X, pd.DataFrame):
            new_X = pd.DataFrame(new_X, columns=self.features.columns)
        scaled_X = self.scale(new_X)
        predicted_labels = []
        for i in range(scaled_X.shape[0]):
            x = scaled_X.iloc[i]
            # computing distances
            temp = self.features.sub(x, axis=1).pow(2).sum(axis=1)
            temp = temp ** (1 / 2)
            # getting k nearest neighbors
            neighbors_idx = temp.argsort(axis=0)[: self.n_neighbors]
            # getting most frequent label
            label = self.labels.iloc[neighbors_idx].mode().iloc[0]
            predicted_labels.append(label)
        return pd.Series(predicted_labels, name="label", index=new_X.index)

    def cross_validate(self, data, n_folds=5):
        data = data.sample(frac=1).reset_index()
        n = data.shape[0]
        acc = []
        for i in range(n_folds):
            start = int(n * i / n_folds)
            end = int(n * (i + 1) / n_folds)
            X, y = data.drop(columns=["label"]), data["label"]
            X_train, X_test = X.drop(index=range(start, end)), X.iloc[start:end]
            y_train, y_test = y.drop(index=range(start, end)), y.iloc[start:end]
            self.fit(X_train, y_train)
            predictions = self.predict(X_test)
            acc.append(accuracy(y_test, predictions))
        return acc


def accuracy(y, predictions):
    return (y == predictions).sum() / predictions.shape[0]


if __name__ == "__main__":
    data = Data()
    X_train, X_test, y_train, y_test = data.train_test_split(test_size=0.2)
    knn = KNN(n_neighbors=5)
    knn.fit(X=X_train, y=y_train)
    predictions = knn.predict(X_test)
    print(accuracy(y_test, predictions))
    # acc = knn.cross_validate(n_folds=5, data = data.data)
    # print(acc)
    pass
