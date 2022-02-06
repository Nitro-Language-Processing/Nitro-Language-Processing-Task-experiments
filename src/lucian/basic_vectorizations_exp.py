
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    data, tag_to_id = get_all_data(change_ner_tags=True)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]
if __name__=='__main__':
    main()