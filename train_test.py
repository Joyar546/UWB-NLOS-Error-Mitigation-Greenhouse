import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
from sklearn.model_selection import train_test_split
import pandas as pd

from tqdm import tqdm

from models import *

from models.eca_resnet import *
from models.se_resnet import *
from models.resnet import *
from models.CNN import *
from models.REMnet import *
from models.multi_scale_ori import *
from models.eca_resnet18 import *

# Extract dataset
dataset = pd.read_pickle(r'dataset\ranging_position_dataset.pkl')

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper parameters
num_epochs = 150
batch_size = 8
learning_rate = 0.001

# Setup Dataset
ds = np.asarray(dataset[['CIR', 'Error']])
train_ds, test_ds = train_test_split(ds, test_size=0.4, random_state=0)
train_dsx = np.vstack(train_ds[:, 0])
train_dsy = np.vstack(train_ds[:, 1])
test_dsx = np.vstack(test_ds[:, 0])
test_dsy = np.vstack(test_ds[:, 1])

data_tensor = torch.from_numpy(train_dsx)
data_tensor = data_tensor.unsqueeze(1).type(torch.FloatTensor)
target_tensor = torch.from_numpy(train_dsy)
train_dataset = TensorDataset(data_tensor, target_tensor)
data_tensor = torch.from_numpy(test_dsx)
data_tensor = data_tensor.unsqueeze(1).type(torch.FloatTensor)
target_tensor = torch.from_numpy(test_dsy)
test_dataset = TensorDataset(data_tensor, target_tensor)

train_ld = DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      shuffle=True)
test_ld = DataLoader(dataset=test_dataset,
                     batch_size=batch_size,
                     shuffle=False)

# select your model

model_name = 'ECAResNet18'

# CNN
model = ConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ECAResNet
# model = ECAResNet(input_channel=1, layers=[1, 1, 1, 1])
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# SENet
# model = SEResNet(input_channel=1, layers=[1, 1, 1, 1])
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ResNet
# model = ResNet(input_channel=1, layers=[1, 1, 1, 1])
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# MSResNet
# model = MSResNet(input_channel=1, layers=[1, 1, 1, 1])
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# REMNet
# model = REMNet()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
#                              betas=[0.9, 0.999], eps=1e-8)

model = model.to(device)
criterion = nn.L1Loss(reduction='mean').to(device)  # sum or mean?

# 改变Learning rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])

start = time.clock()

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    loss_all = 0

    # train
    model.train()
    for (cirs, labels) in tqdm(train_ld):
        cirsV = Variable(cirs.to(device))  # variable：可存储梯度
        labelsV = Variable(labels.to(device))

        outputs = model(cirsV)  # model要改一下
        loss = criterion(outputs, labelsV)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # train accuracy
    model.eval()
    i = 0
    for i, (cirs, labels) in enumerate(train_ld):
        with torch.no_grad():
            cirsV = Variable(cirs.to(device))  # variable：可存储梯度
            labelsV = Variable(labels.to(device))

            outputsV = model(cirsV)
            loss = criterion(outputsV, labelsV)
            loss_all += loss.item()

    print("Training MAE:", (loss_all/i))
    train_mae = round(loss_all / i, 4)
    train_loss[epoch] = train_mae

    # test
    model.eval()
    i = 0
    test_mae = 0
    temp_mae = 0
    outputs = []
    label = []
    for i, (cirs, labels) in enumerate(test_ld):
        with torch.no_grad():
            cirsV = Variable(cirs.to(device))  # variable：可存储梯度
            labelsV = Variable(labels.to(device))

            outputsV = model(cirsV)
            loss = criterion(outputsV, labelsV)
            loss_all += loss.item()
            lab = labelsV.clone().reshape(-1).detach().cpu().numpy()
            out = outputsV.clone().reshape(-1).detach().cpu().numpy()
            label.extend(lab)
            outputs.extend(out)

    print("Test MAE:", (loss_all/i))
    test_mae = round(loss_all / i, 4)
    test_loss[epoch] = test_mae
    # lab_mean = np.mean(label)
    # lab_std = np.std(label, ddof=1)
    # print("True mean:", lab_mean)
    # print("True std:", lab_std)
    # out_mean = np.mean(outputs)
    # out_std = np.std(outputs, ddof=1)
    # print("Test mean:", out_mean)
    # print("Test std:", out_std)

    # sse = np.square(outputs - label).sum()
    # sst = np.square(y_true - y_true.mean()).sum()
    # r2 = 1 - sse/sst


    # if epoch == 0:
    #     temp_mae = test_mae
    # elif test_mae > temp_mae:
    #     torch.save(model, 'saved/' + model_name + '/Train' + str(train_mae) + 'Test' + str(test_mae) + '.pkl')
    #     temp_test = test_mae

end = time.clock()

plt.plot(train_loss)
plt.show()
train_loss = pd.DataFrame(train_loss)
# train_loss.to_csv('saved/' + model_name + '/Train_loss.csv')
test_loss = pd.DataFrame(test_loss)
# test_loss.to_csv('saved/' + model_name + '/Test_loss.csv')

# parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Number of parameters:", parameter)

print('Running time: %s Seconds' % (end-start))
