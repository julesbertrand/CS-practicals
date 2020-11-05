import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from bs4 import BeautifulSoup  # html parser
import string
import re  # punctuation
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer


class SentimentAnalysis:
    TEXT_COL = "review"
    LABEL_COL = "sentiment"

    def __init__(
        self,
        path,
        test_size=0.2,
        seed=None,
        vocab_size=10000,
        mode="count",
        max_len=500,
        trunc_type="post",
    ):
        self.data = pd.read_csv(path)
        self.length = len(self.data.index)
        self.test_size = test_size
        self.set_seed(seed)
        self.vocab_size = vocab_size
        self.mode = mode
        self.max_len = max_len
        self.trunc_type = trunc_type
        self.stopwords = set(stopwords.words("english"))

        if self.mode == "pad":
            self.padder = lambda x: pad_sequences(
                x, maxlen=self.max_len, truncating=self.trunc_type
            )

    def set_seed(self, seed=None):
        self.seed = seed
        if seed:
            np.random.seed(seed=seed)

    def train_test_split(self, test_size=0.1, seed=None):
        self.le = LabelEncoder()
        X = self.data[self.TEXT_COL]
        y = self.le.fit_transform(self.data[self.LABEL_COL])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        return X_train, X_test, y_train, y_test

    # decorator
    def __applymap(func):
        def wrapper(self, text, *args, **kwargs):
            func_name = func.__name__.replace("__", "").replace("_", " ")
            print(" Started {} ".format(func_name).center(100, "="))
            return text.apply(lambda x: func(x, *args, **kwargs))

        return wrapper

    @__applymap
    def __html_parser(text):
        return BeautifulSoup(text, "html.parser").get_text()

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
        res = " ".join(list(map(stemmer, text)))
        return res

    def _preprocess(self, text):
        """
        parse html, remove punctuation, lowercase, remove stopwords, stem
        vectorize and pad to get vectors of size n*self.max_len
        """
        text = self.__html_parser(text)
        text = self.__remove_punctuation(text)
        text = self.__lower(text)

        tokenizer = RegexpTokenizer(r"\w+")
        seq = self.__tokenize(
            text, tokenizer
        )  # convert text to seq to remove stopwords and stem

        seq = self.__remove_stop_words(seq, self.stopwords)
        stemmer = PorterStemmer()
        text = self.__stem(seq, stemmer.stem)  # here text is a string
        return text

    def _vectorize(self, text):
        if self.mode == "pad":
            # approach 1: vectors are sentences padded at self.max_len
            # at the end, data of size n * max_len with oov_token for words not in vocab, sequential
            seq = self.vectorizer.texts_to_sequences(text)
            seq = self.padder(seq)
        elif self.mode in ["binary", "count", "tfidf", "freq"]:
            # approach 2: vectors are list of words with v[i] = j where i is a word index and j the num of occurences of this word
            # data of size n * vocab_size, not sequential
            seq = self.vectorizer.texts_to_matrix(text, mode=self.mode)
        else:
            raise ValueError(
                "{} mode does not exist for Tokenizer.text_to_matrix. mode must be one of {}.".format(
                    self.mode, ", ".join(["pad", "binary", "count", "tfidf", "freq"])
                )
            )
        return seq

    def preprocess(self):
        """ train test split and apply the self._proprocess function """
        training_text, test_text, self.y_train, self.y_test = self.train_test_split(
            test_size=self.test_size, seed=self.seed
        )
        self.train_text = self._preprocess(training_text)
        self.test_text = self._preprocess(test_text)

    def build_model(self, dense_layers=[], mode=None, embedding_dim=10):
        if mode:
            self.mode = mode
        self.vectorizer = Tokenizer(
            num_words=self.vocab_size, filters="", lower=False, oov_token="<OOV>"
        )
        if self.mode == "pad":
            #             inputs = tf.keras.Input(shape=(self.max_len,), name='input')
            inputs = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=embedding_dim,
                input_length=self.max_len,
            )(inputs)
            x = tf.keras.layers.Flatten(name="flatten")(x)
        else:
            inputs = tf.keras.Input(shape=(self.vocab_size,), name="input")
            x = tf.keras.layers.Flatten(name="flatten")(inputs)

        for i, dim in enumerate(dense_layers):
            x = tf.keras.layers.Dense(
                dim, activation="relu", name="dense_{}".format(i + 1)
            )(x)
        y = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_sigmoid")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=y)
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.model.summary()

    def fit_model(self, epochs=10):
        self.vectorizer.fit_on_texts(self.train_text)
        self.X_train = self._vectorize(self.train_text)
        self.X_test = self._vectorize(self.test_text)
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test),
        )

    def predict(self, X):
        X = self._preprocess(X)
        X = self._vectorize(X)
        y_pred = model.predict(X_pad)
        return y_pred


if __name__ == "__main__":
    # path
    PATH = "./IMDB_Dataset.csv"

    # preprocessing
    VOCAB_SIZE = 10000
    MODE = "tfidf"
    MAX_LEN = 500
    TRUNC_TYPE = "post"
    TEST_SIZE = 0.1

    s = SentimentAnalysis(
        path=PATH,
        test_size=TEST_SIZE,
        seed=100,
        vocab_size=VOCAB_SIZE,
        mode=MODE,
        max_len=MAX_LEN,
        trunc_type=TRUNC_TYPE,
    )
    s.preprocess()

    # model related
    EMBEDDING_DIM = 10  # 'pad' only
    DENSE_LAYERS = [6] if MODE is "pad" else [50, 6]
    EPOCHS = 10

    s.build_model(mode=MODE, embedding_dim=EMBEDDING_DIM, dense_layers=DENSE_LAYERS)
    s.fit_model(epochs=EPOCHS)
