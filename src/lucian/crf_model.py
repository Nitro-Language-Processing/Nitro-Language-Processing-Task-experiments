import torch
from TorchCRF import CRF
from statistics import mean
from src.common.util import get_all_data
import torch.nn as an
from numpy import vstack
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

vocab = dict()
MAX_SEQ_LEN = 300
BATCH_SIZE = 32
NUM_LABELS = 16

class LiRoDataset():
    def __init__(self):
        self.X, self.y, self.seq_lens = prepare_data_for_crf()

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx], self.seq_lens[idx])

    def __len__(self):
        return len(self.X)


def build_vocab(datapoints):
    global vocab
    cnt = 1
    for datapoint in datapoints:
        tokens = datapoint["tokens"]
        for token in tokens:
            if token not in vocab:
                vocab[token] = cnt
                cnt += 1

def prepare_subset_for_crf(datapoints):
    global vocab
    X, y = [], []

    token_pad_value = 0
    ner_id_pad_value = 0 # 16
    seq_lens = []

    for datapoint in datapoints:
        ner_ids = datapoint["ner_ids"][:MAX_SEQ_LEN]
        tokens = datapoint["tokens"][:MAX_SEQ_LEN]

        seq_lens.append(len(tokens))

        tokens = [vocab[token] for token in tokens]

        while len(tokens) < MAX_SEQ_LEN:
            tokens.append(token_pad_value)

        while len(ner_ids) < MAX_SEQ_LEN:
            ner_ids.append(ner_id_pad_value)

        X.append(tokens)
        y.append(ner_ids)

    return np.array(X), np.array(y), np.array(seq_lens)


def example():
    batch_size = 2
    sequence_size = 3
    num_labels = 5

    crf = CRF(num_labels)

    mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device)  # (batch_size. sequence_size)
    labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)  # (batch_size, sequence_size)
    hidden = torch.randn((batch_size, sequence_size, num_labels), requires_grad=True).to(device)

    forward_output = crf.forward(hidden, labels, mask)
    print(f"Forward outputs: {forward_output}")

    decoded_predictions = crf.viterbi_decode(hidden, mask)
    print(f"Decoded predictions: {decoded_predictions}")


def data_analysis(datapoints):
    tokens_lens = []
    for datapoint in datapoints:
        tokens = datapoint["tokens"]
        tokens_lens.append(len(tokens))
    print("*" * 10)
    print("MIN: ", min(tokens_lens))
    print("MEAN: ", mean(tokens_lens))
    print("MAX: ", max(tokens_lens))
    print("*" * 10)
    # plt.hist(tokens_lens, bins=500)
    # plt.show()


def prepare_data_for_crf():
    data, _ = get_all_data(change_ner_tags=True, change_ner_ids=True, first_n=100)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

    train_valid = train + valid

    data_analysis(train)
    data_analysis(valid)
    data_analysis(test)
    data_analysis(train + valid + test)

    build_vocab(train + valid + test)
    # X_train, y_train = prepare_subset_for_crf(train_valid)
    # X_test, y_test = prepare_subset_for_crf(test)
    return prepare_subset_for_crf(train + valid + test)

from tqdm import tqdm
import pdb

# def get_class_weight(labels):

def train_model(train_dl, crf_model):
    # define the optimization
    criterion = CrossEntropyLoss()
    num_epochs = 10
    optimizer = Adam(crf_model.parameters())
    # enumerate epochs
    token_pad_value = 0
    ner_id_pad_value = 0  # 16
    for epoch in tqdm(range(num_epochs)):
        # enumerate mini batches
        for i, (inputs, targets, seq_lens) in enumerate(train_dl):
            # mask = torch.zeros_like(inputs).bool()
            # pdb.set_trace()
            # create random array of floats in equal dimension to input_ids
            rand = torch.rand(inputs.shape)
            # where the random array is less than 0.15, we set true
            mask = rand < 0.15
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output

            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            # print(inputs.shape, targets.shape)
            # mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device)  # (batch_size. sequence_size)
            # labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)  # (batch_size, sequence_size)
            hidden = torch.randn((BATCH_SIZE, MAX_SEQ_LEN, NUM_LABELS), requires_grad=True).to(device)
            # pdb.set_trace()
            # print(i, "*" * 50, hidden.shape, targets.shape, mask.shape)
            if hidden.shape[0] != BATCH_SIZE or targets.shape[0] != BATCH_SIZE or mask.shape[0] != BATCH_SIZE:
                continue
            losses = crf_model.forward(h=hidden,
                                     labels=targets,
                                     mask=mask)

            yhat = crf_model.viterbi_decode(hidden, mask)
            # calculate loss
            # pdb.set_trace()

            for j in range(len(yhat)):
                while len(yhat[j]) < MAX_SEQ_LEN:
                    yhat[j].append(token_pad_value)
            yhat = torch.tensor(yhat)

            # pdb.set_trace()
            loss = criterion(torch.transpose(torch.nn.functional.one_hot(yhat), 2, 1).float(), targets)
            loss.requires_grad = True
            # pdb.set_trace()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

        # evaluate the model


def evaluate_model(test_dl, crf_model):
    token_pad_value = 0
    ner_id_pad_value = 0  # 16
    predictions, actuals = list(), list()
    for i, (inputs, targets, seq_lens) in tqdm(enumerate(test_dl)):
        if inputs.shape[0] != BATCH_SIZE or targets.shape[0] != BATCH_SIZE:
            continue
        # evaluate the model on the test set
        print(i, "*" * 50, inputs.shape, targets.shape)
        rand = torch.rand(inputs.shape)
        # where the random array is less than 0.15, we set true
        mask = rand < 0.15
        hidden = torch.randn((BATCH_SIZE, MAX_SEQ_LEN, NUM_LABELS), requires_grad=True).to(device)

        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        yhat = crf_model.viterbi_decode(hidden, mask)
        # retrieve numpy array

        for j in range(len(yhat)):
            while len(yhat[j]) < MAX_SEQ_LEN:
                yhat[j].append(token_pad_value)
        yhat = torch.tensor(yhat)

        yhat = yhat.detach().numpy()
        actual = targets.numpy()

        # pdb.set_trace()
        # actual = actual.reshape((len(actual), 1))
        # # round to class values
        # yhat = yhat.round()
        # store

        flatten_yhat = []
        for j in range(len(yhat)):
            for elem in yhat[j][:seq_lens[j]]:
                flatten_yhat.append(elem)

        flattent_actual = []
        for j in range(len(actual)):
            for elem in actual[j][:seq_lens[j]]:
                flattent_actual.append(elem)

        predictions.append(flatten_yhat)
        actuals.append(flattent_actual)

    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    pdb.set_trace()

    actuals = actuals[0]
    predictions = predictions[0]

    acc = accuracy_score(actuals, predictions)
    return acc


def train_crf_model():
    liro_dataset = LiRoDataset()
    train_size = int(0.8 * len(liro_dataset))
    test_size = len(liro_dataset) - train_size
    train, test = torch.utils.data.random_split(liro_dataset, [train_size, test_size])

    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    # mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device)  # (batch_size. sequence_size)
    # labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)  # (batch_size, sequence_size)
    # hidden = torch.randn((BATCH_SIZE, MAX_SEQ_LEN, NUM_LABELS), requires_grad=True).to(device)

    crf_model = CRF(NUM_LABELS)

    # forward_output = crf_model.forward(hidden, labels, mask)
    # print(f"Forward outputs: {forward_output}")
    #
    # decoded_predictions = crf_model.viterbi_decode(hidden, mask)
    # print(f"Decoded predictions: {decoded_predictions}")

    train_model(train_dl, crf_model)
    acc = evaluate_model(test_dl, crf_model)
    print('Accuracy: %.3f' % acc)


def main():
    global device
    # X_train, y_train, X_test, y_test = prepare_data_for_crf()
    example()
    train_crf_model()

if __name__=="__main__":
    main()