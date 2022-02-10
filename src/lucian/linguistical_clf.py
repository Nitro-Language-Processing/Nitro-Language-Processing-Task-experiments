import gensim
import matplotlib.pyplot as plt
import multiprocessing
import nltk
import numpy as np
import pandas as pd
import pdb
import random
from gensim.models import Word2Vec
# define training data
from gensim.test.utils import common_texts
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.common.util import *


def document_preprocess(document):
    return word_tokenize(document)


def ensemble_voting(X_train, y_train, X_test, y_test):
    estimators = [
        ('svc', SVC(class_weight="balanced")),
        ('random_forest', RandomForestClassifier(class_weight="balanced")),
        ('decision_tree', DecisionTreeClassifier(class_weight="balanced")),
        ('xgb', XGBClassifier())
    ]

    ensemble = Pipeline(steps=[("voter", VotingClassifier(estimators))])
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    print("Ensemble clf score:", f1_score(y_pred, y_test, average="weighted"))


def train_classifier_head(X_train, y_train, X_test, y_test):
    clfs = [
        ('logistic', SVC(class_weight="balanced")),
        ('random_forest', RandomForestClassifier(class_weight="balanced")),
        ('decision_tree', DecisionTreeClassifier(class_weight="balanced")),
        ('xgb', XGBClassifier())
    ]
    for (name, clf) in clfs:
        print(X_train.shape)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        y_pred = np.zeros(shape=y_test.shape)
        test_score = f1_score(y_test, y_pred, average="weighted")
        print("*" * 10)
        print(f"F1 score: {test_score} - CLF: {name}")
        print("*" * 10)


def embed(text, word2vec_model):
    try:
        vector = word2vec_model.wv[document_preprocess(text)]
        vector = np.mean(vector, axis=0)
        vector = np.reshape(vector, (1, vector.shape[0]))
    except KeyError:
        vector = np.random.rand(1, 150)
    return vector


def create_train_test_data(data):
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

    train_valid = train + valid

    vectorized_train = np.array([datapoint["tokens"] for datapoint in train_valid])
    vectorized_test = np.array([datapoint["tokens"] for datapoint in test])
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(len(vectorized_train)):
        vectorized_doc = vectorized_train[i]
        for j in range(len(vectorized_doc)):
            context = " ".join(vectorized_doc[max(0, j - 2): min(len(vectorized_doc) - 1, j + 2)])
            embedded_context = embed(context, word2vec_model)
            # import pdb
            # pdb.set_trace()
            label = train_valid[i]["ner_ids"][j]
            X_train.append(embedded_context)
            y_train.append(label)

    for i in range(len(vectorized_test)):
        vectorized_doc = vectorized_test[i]
        for j in range(len(vectorized_doc)):
            context = " ".join(vectorized_doc[max(0, j - 2): min(len(vectorized_doc) - 1, j + 2)])
            embedded_context = embed(context, word2vec_model)
            label = test[i]["ner_ids"][j]
            X_test.append(embedded_context)
            y_test.append(label)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def classifier_experiment(X_train, y_train, X_test, y_test):
    train_classifier_head(X_train, y_train, X_test, y_test)
    ensemble_voting(X_train, y_train, X_test, y_test)


def main():
    data, _ = get_all_data(first_n=100)
    X_train, y_train, X_test, y_test = create_train_test_data(data)

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    classifier_experiment(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
