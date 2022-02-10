# import torch
# from TorchCRF import CRF
# device = "cuda" if torch.cuda.is_available() else "cpu"
from statistics import mean
from src.common.util import get_all_data

vocab = dict()
MAX_SEQ_LEN = 300
BATCH_SIZE = 32
NUM_LABELS = 16

class LiRoDataset():
    def __init__(self):
        self.X, self.y =

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return len(self.X)


def build_vocab(datapoints):
    global vocab
    cnt = 0
    for datapoint in datapoints:
        tokens = datapoint["tokens"]
        for token in tokens:
            if token not in vocab:
                vocab[token] = cnt
                cnt += 1

def prepare_subset_for_crf(datapoints):
    X, y = [], []
    for datapoint in datapoints:
        ner_ids = datapoint["ner_ids"]
        tokens = datapoint["tokens"]

    return np.array(X), np.array(y)


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

import matplotlib.pyplot as plt

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
    plt.hist(tokens_lens, bins=500)
    plt.show()

def prepare_data_for_crf():
    data, _ = get_all_data()
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

    train_valid = train + valid

    data_analysis(train)
    data_analysis(valid)
    data_analysis(test)
    data_analysis(train + valid + test)
    exit()

    build_vocab(train_valid + test)


    X_train, y_train = prepare_subset_for_crf(train_valid)
    X_test, y_test = prepare_subset_for_crf(test)
    return X_train, y_train, X_test, y_test

# def train_model(train_dl, model):
#     # define the optimization
#     criterion = CrossEntropyLoss()
#     num_epochs = 10
#     optimizer = Adam(model.parameters())
#     # enumerate epochs
#     for epoch in tqdm(range(num_epochs)):
#         # enumerate mini batches
#         for i, (inputs, targets) in enumerate(train_dl):
#             # clear the gradients
#             optimizer.zero_grad()
#             # compute the model output
#             yhat = model(inputs)
#             # calculate loss
#             loss = criterion(yhat, targets)
#             # credit assignment
#             loss.backward()
#             # update model weights
#             optimizer.step()
#
#         # evaluate the model
#
#
# def evaluate_model(test_dl, model):
#     predictions, actuals = list(), list()
#     for i, (inputs, targets) in tqdm(enumerate(test_dl)):
#         # evaluate the model on the test set
#         yhat = model(inputs)
#         # retrieve numpy array
#         yhat = yhat.detach().numpy()
#         actual = targets.numpy()
#         actual = actual.reshape((len(actual), 1))
#         # round to class values
#         yhat = yhat.round()
#         # store
#         predictions.append(yhat)
#         actuals.append(actual)
#     predictions, actuals = vstack(predictions), vstack(actuals)
#     # calculate accuracy
#     acc = accuracy_score(actuals, np.argmax(predictions, axis=-1))
#     return acc
#
#
# # make a class prediction for one row of data
# def predict(row, model):
#     # convert row to data
#     row = Tensor([row])
#     # make prediction
#     yhat = model(row)
#     # retrieve numpy array
#     yhat = yhat.detach().numpy()
#     return yhat

def train_crf_model():
    liro_dataset = LiRoDataset()
    train_size = int(0.8 * len(liro_dataset))
    test_size = len(liro_dataset) - train_size
    train, test = torch.utils.data.random_split(mami_dataset, [train_size, test_size])

    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    for
    mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).to(device)  # (batch_size. sequence_size)
    labels = torch.LongTensor([[0, 2, 3], [1, 4, 1]]).to(device)  # (batch_size, sequence_size)

    hidden = torch.randn((BATCH_SIZE, MAX_SEQ_LEN, NUM_LABELS), requires_grad=True).to(device)

    crf = CRF(NUM_LABELS)

    forward_output = crf.forward(hidden, labels, mask)
    print(f"Forward outputs: {forward_output}")

    decoded_predictions = crf.viterbi_decode(hidden, mask)
    print(f"Decoded predictions: {decoded_predictions}")

    # train_model(train_dl, model)
    # acc = evaluate_model(test_dl, model)
    # print('Accuracy: %.3f' % acc)


def main():
    global device
    X_train, y_train, X_test, y_test = prepare_data_for_crf()
    example()

if __name__=="__main__":
    main()