# Nitro-Language-Processing-Task-experiments
This repository will contain a couple of experiments &amp; benchmarks performed in order to decide the task of the Nitro Language Processing Hackathon

Experiments List:

- current state-of-the-art: Romanian BERT Baseline - 0.86
- ro_core_news_sm default spacy pipeline - Luci - 0.55
- ro_core_news_md default spacy pipeline - Luci - 0.62
- ro_core_news_lg default spacy pipeline - Luci - 0.63
- ro_core_news_sm finetuned spacy pipeline - unassigned - ?
- ro_core_news_md finetuned spacy pipeline - unassigned - ?
- ro_core_news_lg finetuned spacy pipeline - unassigned - ?
- bert embeddings as feature extraction with any classifier over - unassigned - ?
- flair embeddings as feature extraction with any classifier over - unassigned - ?
- fasttext embeddings as feature extraction with any classifier over - unassigned - ?
- word2vec embeddings as feature extraction with any classifier over - Bleo & Luci - ?
- glove embeddings as feature extraction with any classifier over - unassigned - ?
- countvectorizer vectorization as feature extraction with any classifier over - Luci - 0.55
- tfidfvectorizer vectorization as feature extraction with any classifier over - Luci - 0.61
- stringkernels transformation as feature extraction with any classifier over - Luci - ?
- linguistical features as feature extraction with any classifier over - unassigned - ?
- Automated Concatenation of Embeddings for Structured Prediction - https://arxiv.org/pdf/2010.05006.pdf - tonio - ?
- Other BERT-like transformers other than the one from Stefan Dumitrescu's paper - unassigned - ?
- Logistic regression/Perceptron/SVM classifier with context information (let's say 2 tokens to the left, 2 tokens to the right and extract features) - unassigned - ?
- conditional random field (CRF), it is one of the most well known models applied for this NER task (please refer to one of these: https://sklearn-crfsuite.readthedocs.io/en/latest/, https://python-crfsuite.readthedocs.io/en/latest/ ) - unassigned - ?
- SEQ2SEQ model with a LSTM on top - unassigned - ?
- AutoEncoder for feature extraction in the latent space with a classifier on top - unassigned - ?
- Perceiver based architecture with a word embedding in order to extract the byte arrays- unassigned - ?


*NOTE: for the experiments that mention 'any classifier over' please specify the used classifier*
