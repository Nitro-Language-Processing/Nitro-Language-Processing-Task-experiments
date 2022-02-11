# Nitro-Language-Processing-Task-experiments
This repository will contain a couple of experiments &amp; benchmarks performed in order to decide the task of the Nitro Language Processing Hackathon

Experiments List:

- current state-of-the-art: Romanian BERT Baseline - 0.86
- result if just the majority class label is predicted every time - Luci - 0.63
- ro_core_news_sm default spacy pipeline - Luci - 0.55
- ro_core_news_md default spacy pipeline - Luci - 0.62
- ro_core_news_lg default spacy pipeline - Luci - 0.63
- ro_core_news_sm finetuned spacy pipeline - Luci - ?
- ro_core_news_md finetuned spacy pipeline - Luci - ?
- ro_core_news_lg finetuned spacy pipeline - Luci - ?
- bert embeddings as feature extraction with any classifier over - unassigned - ?
- flair embeddings as feature extraction with any classifier over (do note that as of February 2022 there are no existing flair vectors for Romanian language, so take into consideration the fact that you also have to train a flair language model for this task, Flair only supports english, dutch, german, french and spanish) - unassigned - ?
- fasttext embeddings as feature extraction with any classifier over - Luci - 0.64 (single best: RandomForest clf, ensemble best: RF+DT+XGB+SVC+MNB: 0.84)
- word2vec embeddings as feature extraction with any classifier over - Bleo & Luci - 0.65 (best: RandomForest clf) 
- glove embeddings as feature extraction with any classifier over - unassigned - ?
- countvectorizer vectorization as feature extraction with any classifier over - Luci - 0.55
- tfidfvectorizer vectorization as feature extraction with any classifier over - Luci - 0.61
- stringkernels transformation as feature extraction with any classifier over - Luci - ?
- linguistical features as feature extraction with any classifier over - Luci - ?
- Automated Concatenation of Embeddings for Structured Prediction - https://arxiv.org/pdf/2010.05006.pdf - tonio - ?
- Other BERT-like transformers other than the one from Stefan Dumitrescu's paper - unassigned - ?
- Logistic regression/Perceptron/SVM classifier with context information (let's say 2 tokens to the left, 2 tokens to the right and extract features) - unassigned - ?
- conditional random field (CRF), it is one of the most well known models applied for this NER task (please refer to one of these: https://sklearn-crfsuite.readthedocs.io/en/latest/, https://python-crfsuite.readthedocs.io/en/latest/ ) - Luci - ?
- SEQ2SEQ model with a LSTM on top - unassigned - ?
- AutoEncoder for feature extraction in the latent space with a classifier on top - unassigned - ?
- Perceiver based architecture with a word embedding in order to extract the byte arrays- unassigned - ?


*NOTE: for the experiments that mention 'any classifier over' please specify the used classifier*
