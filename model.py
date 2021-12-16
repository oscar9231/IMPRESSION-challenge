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
# feats_phy = []
ind = []
comp = []
warm = []

path_feats = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/feats_participant/'
path_labels = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/labels/'

os.chdir(path_feats)
for file in range(39):
    print(file)
    with open(file + '.csv') as par:
        file_content = csv.reader(par, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            feats_par.append(row[:-1])
            ind.append(row[-1])

os.chdir(path_labels)
for file in range(39):
    print(file)
    with open(file + '.csv') as label:
        file_content = csv.reader(label, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            comp.append(row[0])
            warm.append(row[1])

# print(len(feats_par), len(feats_par[0]))
# print(len(feats_sti), len(feats_sti[0]))
# print(len(comp), len(comp[0]))

range_comp = max(comp) - min(comp) + 1
range_warm = max(warm) - min(warm) + 1
print(range_comp, range_warm)

# separation for test data
feats_train_valid = np.array(feats_par)
# feats_test = np.array(feats_par[44922:])
comp_train_valid = np.array(comp)
# comp_test = np.array(comp[44922:])
warm_train_valid = np.array(warm)
# warm_test = np.array(warm[44922:])

# shuffle data
leng = len(feats_train_valid)
indices = np.arange(leng)
random.shuffle(indices)
feats_train_valid[np.arange(leng)] = feats_train_valid[indices]
comp_train_valid[np.arange(leng)] = comp_train_valid[indices]
warm_train_valid[np.arange(leng)] = warm_train_valid[indices]

# separate training and validation data
feats_train = feats_train_valid[:int(0.8*leng)]
feats_valid = feats_train_valid[int(0.8*leng):]
comp_train = comp_train_valid[:int(0.8*leng)]
comp_valid = comp_train_valid[int(0.8*leng):]
warm_train = warm_train_valid[:int(0.8*leng)]
warm_valid = warm_train_valid[int(0.8*leng):]

# normalization
feats_train -= np.mean(feats_train, axis = 0)
feats_train /= np.std(feats_train, axis = 0)
feats_valid -= np.mean(feats_valid, axis = 0)
feats_valid /= np.std(feats_valid, axis = 0)

print('Data preparation completed!')

# model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=len(feats_sti[0]),
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.lstm1 = nn.LSTM(input_size=len(feats_par[0]),
                            hidden_size=32,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

        self.lstm2 = nn.LSTM(input_size=len(feats_sti[0]),
                            hidden_size=32,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

        self.attn = nn.MultiheadAttention(64, 8, batch_first=True)
        self.dense = nn.Linear(128, 32)
        self.acti = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.out1 = nn.Linear(32, range_comp)
        self.out2 = nn.Linear(32, range_warm)

    def forward(self, input_par, input_sti):
        x_par = self.conv1(input_par)
        x_sti = self.conv1(input_sti)
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x_par, _ = self.lstm1(x_par)
        x_sti, _ = self.lstm2(x_sti)
        x_par, _ = self.attn(x_par, x_par, x_par)
        x_sti, _ = self.attn(x_sti, x_sti, x_sti)
        x_par_sti = self.attn(x_par, x_sti, x_sti)
        x_sti_par = self.attn(x_sti, x_par, x_par)
        x_co = x_par_sti + x_sti_par
        x_co.view(x_co.size(0), -1)
        # x = x.mean(dim=1)  # pooling
        x_co = self.dense(x_co)
        x_co = self.acti(x_co)
        x_co = self.drop(x_co)
        comp = self.out1(x_co)
        warm = self.out2(x_co)
        return comp, warm
#
model = NeuralNet()
# model = model.to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
func = nn.MSELoss()

epochs = 100
epoch = 0
batch_size = 64

# training
while epoch < epochs:
    start = timeit.default_timer()
    print("-----epoch: ", epoch, "-----")
    comp_loss_list_train = []
    comp_loss_list_valid = []
    warm_loss_list_train = []
    warm_loss_list_valid = []
    comp_preds_train = []
    comp_preds_valid = []
    warm_preds_train = []
    warm_preds_valid = []

    print('--training begins--')
    model.train()
    j = 0
    input_par = []
    # input_phy = []
    input_sti = []
    label_comp = []
    label_warm = []
    while j < len(x_train) / batch_size:
        if (j + 1) * batch_size > len(x_train):
            input_values = x_train[j * batch_size:]
            input_labels = y_train[j * batch_size:]
        else:
            input_values = x_train[j * batch_size:(j + 1) * batch_size]
            input_labels = y_train[j * batch_size:(j + 1) * batch_size]
        for row in input_values:
            ind = row[-1]
            input_sti.append(feats_sti[ind])
            input_par.append(row[:-1])
            # input_par.append(row[:-10])
            # input_phy.append(row[-10:-1])
        for row in input_labels:
            label_comp.append(row[0])
            label_warm.append(row[1])

        # loss
        preds_comp, preds_warm = model(input_sti, input_par)
        train_loss_comp = func(pred_comp, label_comp)
        train_loss_warm = func(pred_warm, label_warm)
        comp_loss_list_train.append(train_loss_comp)
        warm_loss_list_train.append(train_loss_warm)
        for i in preds_comp:
            comp_preds_train.append(i)
        for i in preds_warm:
            warm_preds_train.append(i)
        j += 1

        # backprop
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss = 0.5*(comp_loss_list_train/len(comp_loss_list_train)) \
                     + 0.5*(warm_loss_list_train/len(warm_loss_list_train))
        train_loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

    print('--training ends--')

# tomorrow from here

    # testing
    # extract features
    print('--testing begins--')
    model.eval()
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
