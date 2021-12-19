import os
import csv
import torch
import torch.nn as nn
import numpy as np
import random
import timeit
from audtorch.metrics.functional import concordance_cc

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

feats_par = np.array(feats_par, dtype=float)
feats_sti = np.array(feats_sti, dtype=float)
comp = np.array(comp, dtype=int)
warm = np.array(warm, dtype=int)
ind = np.array(ind, dtype=int)

# range_comp = max(comp) - min(comp) + 1
# range_warm = max(warm) - min(warm) + 1
# print(range_comp, range_warm)

## separation for test data
# feats_par = np.array(feats_par)
# feats_test = np.array(feats_par[44922:])
# comp_train_valid = np.array(comp)
# comp_test = np.array(comp[44922:])
# warm_train_valid = np.array(warm)
# warm_test = np.array(warm[44922:])

torch.manual_seed(1)

# shuffle data
leng = len(feats_par)
indices = np.arange(leng)
random.shuffle(indices)
feats_par[np.arange(leng)] = feats_par[indices]
comp[np.arange(leng)] = comp[indices]
warm[np.arange(leng)] = warm[indices]

# separate training and validation data
feats_train = feats_par[:int(0.8*leng)]
feats_valid = feats_par[int(0.8*leng):]
comp_train = comp[:int(0.8*leng)]
comp_valid = comp[int(0.8*leng):]
warm_train = warm[:int(0.8*leng)]
warm_valid = warm[int(0.8*leng):]

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
    labels_comp = []
    labels_warm = []
    while j < len(feats_train) / batch_size:
        if (j + 1) * batch_size > len(feats_train):
            input_values = feats_train[j * batch_size:]
            labels_comp = comp_train[j * batch_size:]
            labels_warm = warm_train[j * batch_size:]
        else:
            input_values = feats_train[j * batch_size:(j + 1) * batch_size]
            labels_comp = comp_train[j * batch_size:(j + 1) * batch_size]
            labels_warm = warm_train[j * batch_size:(j + 1) * batch_size]
        for row in input_values:
            ind = row[-1]
            input_sti.append(feats_sti[ind])
            input_par.append(row[:-1])
            # input_par.append(row[:-10])
            # input_phy.append(row[-10:-1])

        # loss
        preds_comp, preds_warm = model(input_sti, input_par)
        train_loss_comp = func(preds_comp, labels_comp)
        train_loss_warm = func(preds_warm, labels_warm)
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
        train_loss = 0.5*(sum(comp_loss_list_train)/len(comp_loss_list_train)) \
                     + 0.5*(sum(warm_loss_list_train)/len(warm_loss_list_train))
        train_loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()

    print('--training ends--')


    # validation
    # extract features
    print('--validation begins--')
    model.eval()
    input_values = feats_valid
    for row in input_values:
        ind = row[-1]
        input_sti.append(feats_sti[ind])
        input_par.append(row[:-1])
    preds_comp, preds_warm = model(input_sti, input_par)
    valid_loss_comp = func(preds_comp, comp_valid)
    valid_loss_warm = func(preds_warm, warm_valid)

    # compute performance for each epoch
    comp_preds_train = np.array(comp_preds_train)
    warm_preds_train = np.array(warm_preds_train)
    comp_preds_valid = np.array(comp_preds_valid)
    warm_preds_valid = np.array(warm_preds_valid)
    comp_train = np.array(comp_train)
    warm_train = np.array(warm_train)
    comp_valid = np.array(comp_valid)
    warm_valid = np.array(warm_valid)
#     comp_preds_train = [np.argmax(p) for p in comp_preds_train]
#     warm_preds_train = [np.argmax(p) for p in warm_preds_train]
#     comp_preds_valid = [np.argmax(p) for p in comp_preds_valid]
#     warm_preds_valid = [np.argmax(p) for p in warm_preds_valid]

    train_ccc_comp = concordance_cc(comp_preds_train, comp_train)
    train_ccc_warm = concordance_cc(warm_preds_train, warm_train)
    valid_ccc_comp = concordance_cc(comp_preds_valid, comp_valid)
    valid_ccc_warm = concordance_cc(warm_preds_valid, warm_valid)

    train_loss_comp = sum(comp_loss_list_train) / len(comp_loss_list_train)
    train_loss_warm = sum(warm_loss_list_train) / len(warm_loss_list_train)

    print('Epoch:', epoch, '|train_loss_comp: %.4f' % train_loss_comp, '|train_ccc_comp: %.4f' % train_ccc_comp,
          '|train_loss_warm: %.4f' % train_loss_warm, '|train_ccc_warm: %.4f' % train_ccc_warm,
          '|valid_loss_comp: %.4f' % valid_loss_comp, '|valid_ccc_comp: %.4f' % valid_ccc_comp,
          '|valid_loss_warm: %.4f' % valid_loss_warm, '|valid_ccc_warm: %.4f' % valid_ccc_warm)

    epoch += 1
    print('---validation ends---')

    stop = timeit.default_timer()
    print('Time: ', stop - start)

