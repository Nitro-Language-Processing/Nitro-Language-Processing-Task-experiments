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
from stringkernels.kernels import polynomial_string_kernel
from stringkernels.kernels import string_kernel



def string_kernel_training(X_train, y_train, X_val, y_val, kernel_option="string"):
    if kernel_option == "poly":
        model = SVC(kernel=polynomial_string_kernel()) # 0.864
    elif kernel_option == "string":
        model = SVC(kernel=string_kernel()) # 0.88
    else:
        raise Exception(f"Wrong kernel string option {kernel_option}")

    X_train = np.reshape(X_train, newshape=(X_train.shape[0], 1))
    y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1))

    X_val = np.reshape(X_val, newshape=(X_val.shape[0], 1))
    y_val = np.reshape(y_val, newshape=(y_val.shape[0], 1))

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print("String kernel score:", f1_score(y_pred, y_val, average="weighted"))

def main():
    data, tag_to_id = get_all_data(change_ner_tags=True, change_ner_ids=True)
    import pdb
    pdb.set_trace()
    train = data["train"]
    valid = data["valid"]
    test = data["test"]
    # cv char 1,1 -> 0.42
    # cv char 1,2 -> 0.55
    # tfidf char (1, 2) -> 0.61
    # tfidf char (1, 1) -> 0.42

    analyzer = "char"
    ngram_range = (1, 1)
    n = 2500
    # cv = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    cv = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    X_train, X_test, y_train, y_test = [], [], [], []
    use_linguistical_features = True
    from src.lucian.linguistical_feat_extractor import get_paper_features
    X_train_str = []
    X_test_str = []
    for doc in tqdm((train + valid)[:n // 10]):
        tokens = doc["tokens"]
        ner_ids = doc["ner_ids"]
        document = doc["reconstructed_document"]
        for idx, (token, ner_id) in enumerate(zip(tokens, ner_ids)):
            if use_linguistical_features:
                feats, string_feats = get_paper_features(token, document, idx)
                X_train.append(feats)
                X_train_str.append(string_feats)
            else:
                X_train.append(token)
            y_train.append(ner_id)

    for doc in tqdm(test[:n // 40]):
        tokens = doc["tokens"]
        ner_ids = doc["ner_ids"]
        document = doc["reconstructed_document"]
        for idx, (token, ner_id) in enumerate(zip(tokens, ner_ids)):
            if use_linguistical_features:
                feats, string_feats = get_paper_features(token, document, idx)
                X_test.append(feats)
                X_test_str.append(string_feats)
            else:
                X_test.append(token)
            y_test.append(ner_id)

    X_train = np.array(X_train)
    X_train_str = np.array(X_train_str)
    X_test = np.array(X_test)
    X_test_str = np.array(X_test_str)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # string_kernel_training(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))
    if not use_linguistical_features:
        print("before vectorization")
        X_train = cv.fit_transform(X_train).toarray()
        X_test = cv.transform(X_test).toarray()
        print("after vectorization")

    if use_linguistical_features:
        cv = CountVectorizer()
        X_train_str = cv.fit_transform(X_train_str).toarray()
        X_test_str = cv.transform(X_test_str).toarray()

    # model = XGBClassifier()
    model = SVC(class_weight="balanced")

    print("before fitting")

    X_train, y_train = X_train[:n], y_train[:n]

    X_train_str, X_test_str = X_train_str[:n], X_test_str[:n // 4]

    X_test, y_test = X_test[:n // 4], y_test[:n // 4]

    X_train = np.hstack((X_train, X_train_str))
    X_test = np.hstack((X_test, X_test_str))

    print(X_train.shape, X_test.shape)
    model.fit(X_train, y_train)

    print("between fitting and prediction")
    y_pred = model.predict(X_test)
    print("after prediction")
    print(f1_score(y_pred, y_test, average="weighted"))


if __name__ == '__main__':
    main()
