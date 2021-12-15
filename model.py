#!/usr/bin/env python
# coding: utf-8

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import timeit

# data preparation
with open('/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/feats_stimuli/sti.csv') as sti:
    file_content = csv.reader(sti, delimiter=',')
    headers = next(file_content, None)
    feats_sti = list(file_content)

feats_par = []
labels = []

path_feats = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/feats_participant/'
path_labels = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/labels/'

os.chdir(path_feats)
for file in range(40):
    print(file)
    with open(file + '.csv') as par:
        file_content = csv.reader(par, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            feats_par.append(row)

os.chdir(path_labels)
for file in range(40):
    print(file)
    with open(file + '.csv') as label:
        file_content = csv.reader(label, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            labels.append(row)

# print(len(feats_par), len(feats_par[0]))
# print(len(feats_sti), len(feats_sti[0]))
# print(len(labels), len(labels[0]))

x_train = feats_par[:44922]
x_test = feats_par[44923:]
y_train = labels[]
y_test = labels[]

x_train -= np.mean(x_train, axis = 0)
x_train /= np.std(x_train, axis = 0)
x_test -= np.mean(x_test, axis = 0)
x_test /= np.std(x_test, axis = 0)

indices = np.arange(len(x_train))
random.shuffle(indices)
x_train[np.arange(len(x_train))] = x_train[indices]
y_train[np.arange(len(y_train))] = y_train[indices]

print('Data preparation completed!')

# model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=len(feats_sti[0]),
                            out_channels=64,
                            kernel_size=3,
                            stride=2,
                            padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64,32,3,2,0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=200,
                            hidden_size=32,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        self.attn = nn.MultiheadAttention(64, 8, batch_first=True)
        self.dense = nn.Linear(64, 16)
        self.acti = nn.ReLU()
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = self.attn(x, x, x)
        x = x.mean(dim=1)  # pooling
        x = self.dense(x)
        x = self.acti(x)
        x = self.out(x)
        return x
#
#
model = NeuralNet()
# model = model.to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
func = nn.CrossEntropyLoss()

epochs = 100
epoch = 0
batch_size = 64

# training
while epoch < epochs:
    start = timeit.default_timer()
    print("-----epoch: ", epoch, "-----")
    comp_loss_list_train = []
    comp_loss_list_test = []
    warm_loss_list_train = []
    warm_loss_list_test = []
    comp_predictions_train = []
    comp_predictions_test = []
    warm_predictions_train = []
    warm_predictions_test = []

    print('--training begins--')
    model.train()
    j = 0
    while j < len(names_train) / batch_size:
        if (j + 1) * batch_size > len(names_train):
            input_values = feats_train[j * batch_size:]
            emolabels = emolabels_train[j * batch_size:]
        else:
            input_values = feats_train[j * batch_size:(j + 1) * batch_size]
            emolabels = emolabels_train[j * batch_size:(j + 1) * batch_size]

        # ser loss
        pred = ser_model(input_values)
        train_ser_loss = func(pred, torch.tensor(emolabels))
        ser_loss_list_train.append(train_ser_loss.item())
        for i in pred:
            predictions_train.append(i.detach().numpy)
        j += 1

        # backprop
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(ser_model.parameters(), 5.0)
        train_ser_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    print('--training ends--')

    # testing
    # extract features
    print('--testing begins--')
    ser_model.eval()
    k = 0
    while k < len(names_test):
        emo = []
        input_values = feats_test[k]

        # ser loss
        pred = ser_model(input_values.view(1, 300, 400))
        emo.append(emolabels_test[k])
        test_ser_loss = func(pred, torch.tensor(emo))
        ser_loss_list_test.append(test_ser_loss.item())
        predictions_test.append(pred.detach().numpy())
        #         print("test_ser_loss:", test_ser_loss.item())

        k += 1

    # compute performance for each epoch
    predictions_train = np.array(predictions_train)
    predictions_test = np.array(predictions_test)
    emolabels_train = np.array(emolabels_train)
    emolabels_test = np.array(emolabels_test)
    predictions_train = [np.argmax(p) for p in predictions_train]
    acc_train = accuracy_score(emolabels_train, predictions_train)
    predictions_test = [np.argmax(p) for p in predictions_test]
    acc_test = accuracy_score(emolabels_test, predictions_test)

    trainserloss = sum(ser_loss_list_train) / len(ser_loss_list_train)
    testserloss = sum(ser_loss_list_test) / len(ser_loss_list_test)

    print('Epoch:', epoch, '|[train]ser_loss: %.4f' % trainserloss, '|[train]accuracy: %.4f' % acc_train,
          '|[test]ser_loss: %.4f' % testserloss, '|[test]accuracy: %.4f' % acc_test)
    print('test: \n', confusion_matrix(emolabels_test, predictions_test))
    print(classification_report(emolabels_test, predictions_test))

    epoch += 1
    print('---testing ends---')

    stop = timeit.default_timer()
    print('Time: ', stop - start)
