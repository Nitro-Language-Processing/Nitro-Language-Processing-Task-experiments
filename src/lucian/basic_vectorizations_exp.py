import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from statistics import mean
from tqdm import tqdm
from xgboost import XGBClassifier

from src.common.util import *


def main():
    data, tag_to_id = get_all_data(change_ner_tags=True, change_ner_ids=True)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]
    # cv char 1,1 -> 0.42
    # cv char 1,2 -> 0.55
    # tfidf char (1, 2) -> 0.61
    # tfidf char (1, 1) -> 0.42

    analyzer = "char"
    ngram_range = (1, 1)
    n = 5000
    # cv = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    cv = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    X_train, X_test, y_train, y_test = [], [], [], []

    for doc in tqdm((train + valid)[:n // 10]):
        tokens = doc["tokens"]
        ner_ids = doc["ner_ids"]
        for (token, ner_id) in zip(tokens, ner_ids):
            X_train.append(token)
            y_train.append(ner_id)

    for doc in tqdm(test[:n // 40]):
        tokens = doc["tokens"]
        ner_ids = doc["ner_ids"]
        for (token, ner_id) in zip(tokens, ner_ids):
            X_test.append(token)
            y_test.append(ner_id)

    print("before vectorization")
    X_train = cv.fit_transform(X_train).toarray()
    X_test = cv.transform(X_test).toarray()
    print("after vectorization")

    # model = XGBClassifier()
    model = SVC(class_weight="balanced")

    print("before fitting")

    X_train, y_train = X_train[:n], y_train[:n]
    X_test, y_test = X_train[:n // 4], y_train[:n // 4]

    print(len(set(y_train)), len(set(y_test)))

    print(X_train.shape, X_test.shape)
    model.fit(X_train, y_train)

    print("between fitting and prediction")
    y_pred = model.predict(X_test)
    print("after prediction")
    print(f1_score(y_pred, y_test, average="weighted"))


if __name__ == '__main__':
    main()
