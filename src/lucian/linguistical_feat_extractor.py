import csv
import inflect
import linalg
import matplotlib.pyplot as plt
import nltk
import nltk
import pronouncing
import textstat
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from statistics import *
from statistics import mean
from textblob import TextBlob
from wordfreq import word_frequency

money_symbols = ["$", "£", "€", "lei", "RON", "USD", "EURO", "dolari", "lire", "yeni"]
roman_numerals = "XLVDCMI"

inflect = inflect.engine()

from collections import defaultdict

from wordsegment import load as load_wordsegment
from wordsegment import segment

load_wordsegment()

textstat.set_lang("en")

import numpy as np
import os
import pdb
import pdb
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from gensim import corpora
from gensim.models import Word2Vec
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from transformers import pipeline
from typing import List
from xgboost import XGBRegressor
import nltk
from nltk.corpus import wordnet

PAD_TOKEN = "__PAD__"
word2vec_model = Word2Vec.load("checkpointsword2vec.model")

import nltk

import copy

numpy_arrays_path = "data/numpy_data"
# word2vec_model = Word2Vec.load("src/embeddings_train/fasttext.model")
from nltk.corpus import wordnet

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
GOOD = 0
ERRORS = 0
# TODO change these values
LEFT_LEFT_TOKEN = -4
LEFT_TOKEN = -3
RIGHT_TOKEN = -1
RIGHT_RIGHT_TOKEN = -2

all_languages = set(list(np.save(file="data/all_languages_list.npy", allow_pickle=True)))


def is_there_a_language(text):
    for lang in all_languages:
        if lang in text:
            return True
    return False


def might_be_feminine_surname(text):
    text = text.lower()
    return text.endswith("ei") or text.endswith("a")


import spacy

nlp = spacy.load('ro_core_news_sm')
all_stopwords = set(list(nlp.Defaults.stop_words))


def get_stopwords_pct(text):
    tokens = set(word_tokenize(text))
    return len(tokens.intersection(all_stopwords)) / len(tokens)


def count_letters(target):
    return len(target)


def get_phrase_len(phrase):
    return len(phrase)


def get_num_pos_tags(sentence, tokens=None):
    tokens = word_tokenize(sentence) if tokens == None else tokens
    pos_tags = nltk.pos_tag(tokens)
    pos_tags = [pos_tag[1] for pos_tag in pos_tags]
    return len(set(pos_tags)) / len(tokens)


def get_word_position_in_phrase(phrase, start_offset):
    return start_offset / len(phrase)


def get_phrase_num_tokens(phrase):
    return len(word_tokenize(phrase))


def has_money_tag(text):
    global money_symbols
    for sym in money_symbols:
        if sym.lower() in text.lower():
            return True
    return False


def starts_with_capital_letter(word):
    if word[0] in string.ascii_uppercase:
        return True
    return False


def get_len(text):
    return len(text)


def get_capital_letters_pct(text):
    return len([c for c in text if c in string.ascii_uppercase]) / len(text)


def get_roman_numerals_pct(text):
    global roman_numerals
    return len([c for c in text if c in roman_numerals]) / len(text)


def get_digits_pct(text):
    return len([c for c in text if c in string.digits]) / len(text)


def get_punctuation_pct(text):
    return len([c for c in text if c in string.punctuation]) / len(text)


def get_dash_pct(text):
    return len([c for c in text if c == "-"]) / len(text)


def get_spaces_pct(text):
    return len([c for c in text if c == " "]) / len(text)


def get_slashes_pct(text):
    return len([c for c in text if c == "/" or c == "\\"]) / len(text)


def get_text_similarity(text_1, text_2):
    pass


def get_dots_pct(text):
    return len([c for c in text if c == "."]) / len(text)


def count_capital_words(text, tokens=None):
    tokens = word_tokenize(text) if tokens == None else tokens
    return sum(map(str.isupper, tokens))


def count_punctuations(text):
    punctuations = """}!"#/$%'(*]+,->.:);=?&@\^_`{<|~["""
    d = dict()
    res = []
    for i in punctuations:
        res.append(text.count(i))
    if len(res):
        return mean(res)
    return 0.0


def get_word_frequency(target, tokens=None):
    tokens = word_tokenize(target) if tokens == None else tokens
    return mean([word_frequency(token, 'ro') for token in tokens])


def get_part_of_speech(sentence, tokens=None):
    tokens = word_tokenize(sentence) if tokens == None else tokens
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([pos_tag[1] for pos_tag in pos_tags])


def get_good_vectorizer():
    return TfidfVectorizer(analyzer='char_wb', n_gram_range=(1, 4))


from nltk.tokenize import TreebankWordTokenizer as twt


def count_sws(text, tokens=None):
    if tokens == None:
        tokens = word_tokenize(text)
    return len([tok for tok in tokens if tok.lower() in stop_words])


def get_sws_pct(text):
    tokens = word_tokenize(text)
    return count_sws(text, tokens) / len(tokens)


def get_paper_features(phrase, target, start_offset, end_offset):
    global ERRORS, GOOD
    context_tokens = get_context_tokens(phrase, start_offset, end_offset)
    if context_tokens == None:
        context_tokens = []
    word = target

    target_ = target

    num_features = []

    for target in [target_]:
        word = target
        num_features_ = [count_letters(target),
                         count_consonants(target),
                         count_vowels(target),
                         get_vowel_pct(target),
                         get_consonants_pct(target),
                         get_double_consonants_pct(target),
                         count_word_senses(target, tokens=word_tokenize(target)),
                         mean([count_word_senses(context_tok) for context_tok in context_tokens]),
                         get_base_word_pct(target, tokens=word_tokenize(word)),
                         has_suffix(target, tokens=word_tokenize(word)),
                         count_letters(target),
                         get_base_word_pct_stem(target, tokens=word_tokenize(word)),
                         has_both_affixes_stem(target, tokens=word_tokenize(word)),
                         count_hypernyms(target, tokens=word_tokenize(word)),
                         count_hyponyms(target, tokens=word_tokenize(word)),
                         count_antonyms(target, tokens=word_tokenize(word)),
                         count_definitions_average_tokens_length(target, tokens=word_tokenize(word)),
                         count_definitions_average_characters_length(target, tokens=word_tokenize(word)),
                         count_definitions_tokens_length(target, tokens=word_tokenize(word)),
                         count_total_phonemes_per_pronounciations(target, tokens=word_tokenize(word)),
                         get_word_frequency(target, tokens=word_tokenize(word)),
                         get_average_syllable_count(target),
                         check_word_compounding(target),
                         get_base_word_pct(target),
                         ]
        for feature in num_features_:
            num_features.append(feature)

    return num_features


if __name__ == '__main__':
    phrase = "Both China and the Philippines flexed their muscles on Wednesday."
    start_offset = 56 + len("Wednesday")
    end_offset = 56 + len("Wednesday")
    print(phrase[start_offset: end_offset])

    for synset in wn.synsets('flexed'):
        print(synset)
        print(dir(synset))
        for lemma in synset.lemmas():
            print(lemma.name())
