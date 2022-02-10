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


def count_letters(target):
    return len(target)


def get_base_word_pct(initial_word, root=None, tokens=None):
    tokens = word_tokenize(initial_word) if tokens == None else tokens
    ans = []
    """the higher the more complex a word it is because it requires many subwords"""
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(1 - (len(root) / len(token)))
    if len(ans):
        return mean(ans)
    return ans


def has_prefix(initial_word, root=None, tokens=None):
    tokens = word_tokenize(initial_word) if tokens == None else tokens
    ans = []
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(not token.startswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_suffix(initial_word, root=None, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(not initial_word.endswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_both_affixes(initial_word, root=None, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        root = lemmatizer.lemmatize(token) if root is None else root
        ans.append(has_prefix(initial_word, root) and has_suffix(initial_word, root))
    if len(ans):
        return mean(ans)
    return ans


def get_base_word_pct_stem(initial_word, root=None, tokens=None):
    """the higher the more complex a word it is because it requires many subwords"""
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(1 - (len(root) / len(initial_word)))
    if len(ans):
        return mean(ans)
    return ans


def has_prefix_stem(initial_word, root=None, tokens=None):
    tokens = word_tokenize(initial_word) if tokens == None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(not initial_word.startswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_suffix_stem(initial_word, root=None, tokens=None):
    tokens = word_tokenize(initial_word) if tokens == None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(not initial_word.endswith(root))
    if len(ans):
        return mean(ans)
    return ans


def has_both_affixes_stem(initial_word, root=None, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        root = stemmer.stem(token) if root is None else root
        ans.append(has_prefix_stem(initial_word, root) and has_suffix_stem(initial_word, root))
    if len(ans):
        return mean(ans)
    return ans


def count_hypernyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        word_synsets = wn.synsets(token)
        ans.append(sum([len(word_synset.hypernyms()) for word_synset in word_synsets]))
    if len(ans):
        return mean(ans)
    return ans


def count_hyponyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        word_synsets = wn.synsets(token)
        ans.append(sum([len(word_synset.hyponyms()) for word_synset in word_synsets]))
    if len(ans):
        return mean(ans)
    return ans


def count_antonyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        antonyms_list = []
        for syn in wordnet.synsets(token):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms_list.append(l.antonyms()[0].name())
        ans.append(len(set(antonyms_list)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_synonyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        synonyms_list = []
        for syn in wordnet.synsets(token):
            for l in syn.synonyms():
                synonyms_list.append(l.name())
        ans.append(len(set(synonyms_list)))
    if len(ans):
        return mean(ans)
    return 0.0


def count_meronyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.meronyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_part_meroynms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.part_meronyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_substance_meroynms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.substance_meronyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_holonyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.holonyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_part_holonyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.part_holonyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_substance_holonyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.substance_holonyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_entailments(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.entailments()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_troponyms(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.troponyms()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_average_tokens_length(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        try:
            res = mean([len(word_tokenize(syn.definition())) for syn in wordnet.synsets(token)])
        except:
            res = 0.0
        ans.append(res)
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_average_characters_length(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        try:
            res = mean([len(syn.definition()) for syn in wordnet.synsets(token)])
        except:
            res = 0.0
        ans.append(res)
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_tokens_length(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(word_tokenize(syn.definition())) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def count_definitions_characters_length(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(sum([len(syn.definition()) for syn in wordnet.synsets(token)]))
    if len(ans):
        return mean(ans)
    return 0.0


def is_singular(word):
    return int(inflect.singular_noun(word))


def is_plural():
    return int(inflect.plural_noun(word))


def check_word_compounding(word, tokens=None):
    tokens = word_tokenize(word) if tokens == None else tokens
    ans = []
    for token in tokens:
        ans.append(len(segment(token)))
    if len(ans):
        return mean(ans)
    return 0.0


def word_origin(word):
    import ety
    origins = ety.origins(word)
    predominant_languages = ["french", "english", "german", "latin", "spanish", "italian", "russian", "greek"]
    mapping = {lang: 0 for lang in predominant_languages}

    origin_languages = []
    for origin in origins:
        for language in predominant_languages:
            if language in origin.language.lower():
                mapping[language] += 1
                break
        mapping[language] = origin.language.lower()

    return max(mapping, key=mapping.get)


def word_polarity(word):
    blob = TextBlob(word)
    return blob.sentiment.polarity


def get_target_phrase_ratio(phrase, word):
    return len(word) / len(phrase)


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


def custom_wup_similarity(word1, word2):
    try:
        syn1 = wordnet.synsets(word1)[0]
        syn2 = wordnet.synsets(word2)[0]
        return syn1.wup_similarity(syn2)
    except:
        return 0.0


def get_wup_avg_similarity(target, tokens=None):
    tokens = word_tokenize(target) if tokens is not None else tokens
    if len(tokens) == 1:
        return 0.0
    else:
        ans = []
        for tok in tokens:
            for tok_ in tokens:
                ans.append(custom_wup_similarity(tok, tok_))
        if len(ans):
            return mean(ans)
        return 0.0


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


def count_vowels(word):
    return len([c for c in word if c in "aeiou"])


def count_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    return len([c for c in word if c in consonants])


def count_double_consonants(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    cnt = 0
    for i in range(len(word) - 1):
        if word[i] == word[i + 1] and word[i] in consonants and word[i + 1] in consonants:
            cnt += 1
    return cnt


def get_double_consonants_pct(word):
    return count_double_consonants(word) / len(word)


def get_vowel_pct(word):
    return count_vowels(word) / len(word)


def get_consonants_pct(word):
    return count_consonants(word) / len(word)


def get_part_of_speech(sentence, tokens=None):
    tokens = word_tokenize(sentence) if tokens == None else tokens
    pos_tags = nltk.pos_tag(tokens)
    return " ".join([pos_tag[1] for pos_tag in pos_tags])


def get_good_vectorizer():
    return TfidfVectorizer(analyzer='char_wb', n_gram_range=(1, 4))


from nltk.tokenize import TreebankWordTokenizer as twt


def spans(phrase):
    return list(twt().span_tokenize(phrase))


stop_words = set(stopwords.words('english'))


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
