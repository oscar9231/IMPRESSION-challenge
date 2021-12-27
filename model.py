import os
import csv
import torch
import random
import timeit
import numpy as np
import torch.nn as nn
import torch.optim as optim
from audtorch.metrics.functional import concordance_cc
from torch.utils.data import TensorDataset, DataLoader

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
for file in range(40):
#     print(file)
    with open(str(file) + '.csv') as par:
        file_content = csv.reader(par, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            feats_par.append(row[:-1])
            ind.append(row[-1])

os.chdir(path_labels)
for file in range(40):
#     print(file)
    with open(str(file) + '.csv') as label:
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
comp = np.array(comp, dtype=float)
warm = np.array(warm, dtype=float)
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
np.seterr(divide='ignore', invalid='ignore') # avoid "divide by zero" or "divide by NaN"
feats_train -= np.mean(feats_train, axis = 0)
feats_train /= (np.std(feats_train, axis = 0) + 0.001)
feats_valid -= np.mean(feats_valid, axis = 0)
feats_valid /= (np.std(feats_valid, axis = 0) + 0.001)

feats_train = torch.from_numpy(feats_train)
feats_valid = torch.from_numpy(feats_valid)
feats_sti = torch.from_numpy(feats_sti)
ind_train = torch.from_numpy(ind_train)
ind_valid = torch.from_numpy(ind_valid)
comp_train = torch.from_numpy(comp_train)
comp_valid = torch.from_numpy(comp_valid)
warm_train = torch.from_numpy(warm_train)
warm_valid = torch.from_numpy(warm_valid)

print(feats_train.size(), feats_valid.size(), feats_sti.size())
print(ind_train.size(), ind_valid.size())
print(comp_train.size(), comp_valid.size(), warm_train.size(), warm_valid.size())

trainset = TensorDataset(feats_train, ind_train, comp_train, warm_train)
validset = TensorDataset(feats_valid, ind_valid, comp_valid, warm_valid)
traindata = DataLoader(dataset=trainset, batch_size=64, shuffle=False)
validdata = DataLoader(dataset=validset, batch_size=64, shuffle=False)

print('Data preparation completed!')


# model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=len(feats_par[0]),
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
        self.conv2 = nn.Sequential(
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
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=len(feats_sti[0]),
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.attn = nn.MultiheadAttention(128, 8, batch_first=True)
        self.dense = nn.Linear(128, 16)
        self.acti = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(16, 1)

    def forward(self, input_par, input_sti):
        # lstm
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x_par, _ = self.lstm1(input_par)
        x_sti, _ = self.lstm2(input_sti)
        # attention
        x_par, _ = self.attn(x_par, x_par, x_par)
        x_sti, _ = self.attn(x_sti, x_sti, x_sti)
        x_par_sti, _ = self.attn(x_par, x_sti, x_sti)
        x_sti_par, _ = self.attn(x_sti, x_par, x_par)
        # concatenation
        x_co = torch.cat((x_par, x_sti, x_par_sti, x_sti_par), 1)
#         x_co = x_co.view(x_co.size(0), -1)
        x_co = x_co.mean(dim=1)  # pooling
        x_co = self.dense(x_co)
        x_co = self.acti(x_co)
        x_co = self.drop(x_co)
        comp = self.out(x_co)
        warm = self.out(x_co)
        return comp, warm

model = NeuralNet()
model = model.to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
func = nn.MSELoss()

# training
for epoch in range(100):
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
    for input_par, inds, labels_comp, labels_warm in traindata:
        input_sti = torch.tensor([])
        for inde in inds:
            input_sti = torch.cat((input_sti, feats_sti[inde]), 0)
        input_par = input_par.reshape(input_par.shape[0], 1, input_par.shape[1])
        input_sti = input_sti.reshape(input_par.shape[0], 1, -1)    
        # loss
        preds_comp, preds_warm = model(input_par, input_sti)
        train_loss_comp = func(preds_comp.squeeze(), labels_comp)
        train_loss_warm = func(preds_warm.squeeze(), labels_warm)
        comp_loss_list_train.append(train_loss_comp.item())
        warm_loss_list_train.append(train_loss_warm.item())
        for i in preds_comp:
            comp_preds_train.append(i.item())
        for i in preds_warm:
            warm_preds_train.append(i.item())
        # backprop
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss = 0.5*train_loss_comp + 0.5*train_loss_warm
        train_loss.backward()
        optimizer.step()

    print('--training ends--')

    # validation
    print('--validation begins--')
    model.eval()
    input_par = feats_valid
    for input_par, inds, labels_comp, labels_warm in validdata:
        input_sti = torch.tensor([])
        for inde in inds:
            input_sti = torch.cat((input_sti, feats_sti[inde]), 0)
        input_par = input_par.reshape(input_par.shape[0], 1, input_par.shape[1])
        input_sti = input_sti.reshape(input_par.shape[0], 1, -1)
        # loss
        preds_comp, preds_warm = model(input_par, input_sti)
        valid_loss_comp = func(preds_comp.squeeze(), labels_comp)
        valid_loss_warm = func(preds_warm.squeeze(), labels_warm)
        comp_loss_list_valid.append(valid_loss_comp.item())
        warm_loss_list_valid.append(valid_loss_warm.item())
        for i in preds_comp:
            comp_preds_valid.append(i.item())
        for i in preds_warm:
            warm_preds_valid.append(i.item())

    # compute performance for each epoch
    comp_preds_train = torch.tensor(comp_preds_train)
    warm_preds_train = torch.tensor(warm_preds_train)
    comp_preds_valid = torch.tensor(comp_preds_valid)
    warm_preds_valid = torch.tensor(warm_preds_valid)

    train_ccc_comp = concordance_cc(comp_preds_train, comp_train)
    train_ccc_warm = concordance_cc(warm_preds_train, warm_train)
    valid_ccc_comp = concordance_cc(comp_preds_valid, comp_valid)
    valid_ccc_warm = concordance_cc(warm_preds_valid, warm_valid)
    
    train_loss_comp = sum(comp_loss_list_train) / len(comp_loss_list_train)
    train_loss_warm = sum(warm_loss_list_train) / len(warm_loss_list_train)
    valid_loss_comp = sum(comp_loss_list_valid) / len(comp_loss_list_valid)
    valid_loss_warm = sum(warm_loss_list_valid) / len(warm_loss_list_valid)
        
    print('train_loss_comp: %.4f' % train_loss_comp, '|train_ccc_comp: %.4f' % train_ccc_comp, '\n'
          'train_loss_warm: %.4f' % train_loss_warm, '|train_ccc_warm: %.4f' % train_ccc_warm, '\n'
          'valid_loss_comp: %.4f' % valid_loss_comp, '|valid_ccc_comp: %.4f' % valid_ccc_comp, '\n'
          'valid_loss_warm: %.4f' % valid_loss_warm, '|valid_ccc_warm: %.4f' % valid_ccc_warm)

    print('---validation ends---')

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    scheduler.step()
    
    
# # training
# epoch = 100
# batch_size = 64

# while epoch < epochs:
#     start = timeit.default_timer()
#     print("-----epoch: ", epoch, "-----")
#     comp_loss_list_train = []
#     comp_loss_list_valid = []
#     warm_loss_list_train = []
#     warm_loss_list_valid = []
#     comp_preds_train = []
#     comp_preds_valid = []
#     warm_preds_train = []
#     warm_preds_valid = []

#     print('--training begins--')
#     model.train()
#     j = 0
#     while j < len(feats_train) / batch_size:
#         # input_phy = []
#         input_sti = torch.tensor([])
#         if (j + 1) * batch_size > len(feats_train):
#             input_par = feats_train[j * batch_size:]
#             labels_comp = comp_train[j * batch_size:]
#             labels_warm = warm_train[j * batch_size:]
#             ind_fusion = ind_train[j * batch_size:]
#         else:
#             input_par = feats_train[j * batch_size:(j + 1) * batch_size]
#             labels_comp = comp_train[j * batch_size:(j + 1) * batch_size]
#             labels_warm = warm_train[j * batch_size:(j + 1) * batch_size]
#             ind_fusion = ind_train[j * batch_size:(j + 1) * batch_size]
#         for inde in ind_fusion:
#             input_sti = torch.cat((input_sti, feats_sti[inde]), 0)
#             # input_par.append(feats_sti[ind])
#             # input_par.append(row[:-10])
#             # input_phy.append(row[-10:-1])
#         input_par = torch.tensor(input_par).reshape(input_par.shape[0], 1, input_par.shape[1])
#         input_sti = torch.tensor(input_sti).reshape(input_par.shape[0], 1, -1)
#         print(input_par.size(), input_sti.size())

#         # loss
#         preds_comp, preds_warm = model(input_par, input_sti)
# #         print(preds_comp.detach(), preds_warm.detach())
#         train_loss_comp = func(preds_comp, torch.tensor(labels_comp))
#         train_loss_warm = func(preds_warm, torch.tensor(labels_warm))
#         comp_loss_list_train.append(train_loss_comp.item())
#         warm_loss_list_train.append(train_loss_warm.item())
#         for i in preds_comp:
#             comp_preds_train.append(i.detach().numpy())
#         for i in preds_warm:
#             warm_preds_train.append(i.detach().numpy())
#         j += 1

#         # backprop
#         optimizer.zero_grad()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         train_loss = 0.5*train_loss_comp + 0.5*train_loss_warm
# #         print(j, train_loss.item())
#         train_loss.backward()
#         optimizer.step()
#         # torch.cuda.empty_cache()

#     print('--training ends--')
