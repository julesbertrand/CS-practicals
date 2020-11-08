"""
Toxicity classifier implementation
Using jigsaw-unintended-bias-in-toxicity-classification
Install kaggle and download data first
"""
# import os
# import zipfile
# os.system(
#     "kaggle competitions download -f train.csv jigsaw-unintended-bias-in-toxicity-classification -p ./data"
# )
# with zipfile.ZipFile("./data/train.csv.zip","r") as zip_ref:
#     zip_ref.extractall("./data/")
# os.remove("./data/train.csv.zip")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA, TruncatedSVD


class ToxicityClf:
    def __init__(self, path, labels="target", text="comment_text", seed=None):
        self._params = {}
        print(" Opening data ".center(100, "="))
        self.data = pd.read_csv(path)
        self.data.rename(columns={labels: "target", text: "comment_text"}, inplace=True)
        print(" Data opened ".center(100, "="))
        self.set_seed(seed)

    def set_seed(self, seed=None):
        self._params["seed"] = seed
        if seed:
            np.random.seed(seed)

    def get_params(self):
        return self._params

    def set_params(self, params):
        for key, val in params:
            if key in self._params.keys():
                self._params[key] = val
            else:
                raise KeyError("This param does not exist: {}".format(key))

    def subset_data(self, category, importance_threshold):
        """
        Given a category and a threshold, will reduce self.data to
            - rows where the category score is > threshold
            - columns with labels, and text, and this category
        """
        self._params["subset_category"] = category
        self._params["subset_threshold"] = importance_threshold
        self.data = self.data[["target", "comment_text", category]]
        old_length = len(self.data.index)
        self.data = self.data[
            ~np.isnan(self.data[category]) & self.data[category] > importance_threshold
        ]
        new_length = len(self.data.index)
        print(
            "Had {} datapoints, removed {}, remaining {}".format(
                old_length, old_length - new_length, new_length
            )
        )

    def print_distrib_toxicity_score(self, col, importance_threshold):
        """ print histogram of distribution of toxicity for the choosen column """
        plt.hist(self.data[self.data[col] > importance_threshold].target)
        title = "toxicity score distribution for data about {}".format(col)
        plt.title(title)
        plt.show()

    def balance_data(self, start=0, end=1, step=0.05):
        """
        Print an evolution of % of positive class in dataset given thereshold
        Then the user can choose what threshold to use to compute the classes (toxic /non toxic)
        """
        thresholds = np.arange(start, end, step)
        toxic_rates = [np.mean(self.data.target > t) for t in thresholds]
        for i, t in enumerate(thresholds):
            print(
                "% of comments with toxicity scores > {:.2f}: {:.1f}%".format(
                    t, toxic_rates[i] * 100
                )
            )
        plt.plot(thresholds, toxic_rates)
        plt.title("% of comments with toxicity score > t")
        plt.show()
        while True:
            t = input("Choose threshold above which comment will be considered toxic:")
            if not 0 < float(t) < 1:
                print("Invalid Input: need to be float > 0 and < 1.")
            else:
                self._params["clf_threshold"] = float(t)
                break
        toxic_rate = np.mean(self.data.target > self._params["clf_threshold"])
        print(
            "% of comments with toxicity scores > {:.2f}: {:.1f}%".format(
                self._params["clf_threshold"], toxic_rate * 100
            )
        )
        self.data["toxic_cls"] = self.data.target.values > self._params["clf_threshold"]

    # decorator
    def __applymap(func):
        def wrapper(self, text, *args, **kwargs):
            func_name = func.__name__.replace("__", "").replace("_", " ")
            print(" Currently: {} ".format(func_name))
            return text.apply(lambda x: func(x, *args, **kwargs))

        return wrapper

    @__applymap
    def __remove_punctuation(text):
        return re.sub(r"[^\w\s]", "", text)

    @__applymap
    def __lower(text):
        return text.lower()

    @__applymap
    def __tokenize(text, tokenizer):
        return tokenizer.tokenize(text)

    @__applymap
    def __remove_stop_words(text, stopwords):
        return [w for w in text if w not in stopwords]

    @__applymap
    def __stem(text, stemmer):
        res = " ".join(list(map(stemmer.stem, text)))
        return res

    def _preprocess(self, text):
        """
        remove punctuation, lowercase, remove stopwords, stem
        """
        print(" Preprocessing ".center(100, "="))
        text = self.__remove_punctuation(text)
        text = self.__lower(text)
        seq = self.__tokenize(
            text, RegexpTokenizer(r"\w+")
        )  # convert text to seq to remove stopwords and stem
        seq = self.__remove_stop_words(seq, set(stopwords.words("english")))
        text = self.__stem(seq, PorterStemmer())  # here text is a string
        print(" Preprocessing: Done ".center(100, "="))
        return text

    def preprocess(self, test_size=0.2, vocab_size=5000):
        """
        preprocess data (see self._preprocess)
        train test split (sklearn)
        vectorize using TfidfVectorizer with vocab_size as max_features
        """
        self._params["test_set_size"] = test_size
        self._params["vocab_size"] = vocab_size
        self.text_preprocessed = self._preprocess(self.data["comment_text"])
        train_text, test_text, self.y_train, self.y_test = train_test_split(
            self.text_preprocessed,
            self.data["toxic_cls"],
            test_size=test_size,
            random_state=self._params["seed"],
        )
        self.vectorizer = TfidfVectorizer(
            lowercase=False,  # already done
            stop_words=None,  # already done
            max_features=vocab_size,  # max vocab size
        )
        self.X_train = self.vectorizer.fit_transform(train_text).toarray()
        self.X_test = self.vectorizer.transform(test_text).toarray()

    def pca(self, n_components):
        """ Run PCA """
        self._params["pca_n_components"] = n_components
        self.pca = PCA(random_state=self.seed, n_components=n_components)
        self.pca = self.pca.fit(self.X_train)

    def build_model(self, model_type="", params={}):
        """ Instanciate estimator or build NN if model_type is "NN" """
        if model_type == "LogisticRegression":
            self.model = LogisticRegression(
                random_state=self._params["seed"], max_iter=500, **params
            )
        else:
            raise NotImplementedError("This model is not available")
        self._params["model_type"] = self.model.__class__.__name__

    def fit(self):
        """ fit the model """
        print(" Fitting model: {}".format(self._params["model_type"]).center(100, "="))
        self.model.fit(
            self.X_train,
            self.y_train,
        )
        print(" Model fitted ".center(100, "="))

    def evaluate_model(self, X=None, y=None, preprocess=True):
        """
        Evaluate the model: print classif report and roc auc curve
        Input: X, y: if not given, then evaluate based on internal X_test, y_test from train test split in preprocessing
                if given , you can choose to preprocess or not using the preprocess argument
        Output: classif report from sklearn
        """
        if not self.model:
            raise NotImplementedError("Need to fit a model first")
        if X is None or y is None:
            report = self.classification_report(
                self.X_test, self.y_test, preprocess=False
            )
            self.plot_roc_curve(self.X_test, self.y_test, preprocess=False)
        else:
            report = self.classification_report(X, y, preprocess=preprocess)
            self.plot_roc_curve(X, y, preprocess=preprocess)
        return report

    def predict(self, X, proba=False, preprocess=True):
        """predict toxicity based on threshold given before
        Input: data X, predict proba or not (default: False i.e. classes), preprocess or not (default True)
        Output: predictions or probabilities
        """
        if preprocess:
            X = self._preprocess(X)
            X = self.vectorizer.transform(X).toarray()
        if proba:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def plot_roc_curve(self, X, y, preprocess=True):
        """ predict and print roc curve """
        y_proba = self.predict(X, proba=True, preprocess=preprocess)
        y_true = y
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, c="olive", label="ROC curve (area = {:0.2f})".format(roc_auc))
        plt.plot([0, 1], [0, 1], color="b", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve for {}".format(self._params["model_type"]))
        plt.legend()
        plt.show()

    def classification_report(self, X, y, preprocess=True):
        """ predict and display/output a classif report """
        y_pred = self.predict(X, preprocess=preprocess)
        y_true = y
        report = classification_report(y_true, y_pred)
        print(" Classification Report: ".center(100, "=") + "\n")
        print(report)
        return report


if __name__ == "__main__":
    T = ToxicityClf(path="./data/train.csv", seed=100)
    T.print_distrib_toxicity_score("female", 0.1)
    T.subset_data("female", 0.1)  # focus on commenst where female score > .1
    T.balance_data()
    T.preprocess()
    T.build_model("LogisticRegression")
    T.fit()
    T.evaluate_model()
