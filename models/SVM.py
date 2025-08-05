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

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# Extract dataset
dataset = pd.read_pickle(r'..\dataset\private\ranging_position_dataset.pkl')
# Setup Dataset
ds = np.asarray(dataset[['CIR', 'Error']])
train_ds, test_ds = train_test_split(ds, test_size=0.4, random_state=90)
X = np.vstack(ds[:, 0])
y = np.vstack(ds[:, 1])
X_train = np.vstack(train_ds[:, 0])
y_train = np.vstack(train_ds[:, 1])
X_test = np.vstack(test_ds[:, 0])
y_test = np.vstack(test_ds[:, 1])

start = time.clock()

pipe_SVM = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
pipe_SVM.fit(X_train, y_train)

end = time.clock()

y_pred = pipe_SVM.predict(X_test)
MAE = metrics.mean_absolute_error(y_test, y_pred)  # 平均绝对误差
MSE = metrics.mean_squared_error(y_test, y_pred)  # 均方差
RMSE = math.sqrt(MSE)
r2_score = metrics.r2_score(y_test, y_pred)  # 判定系数,越靠近1越好
out_mean = np.mean(y_pred)
out_std = np.std(y_pred, ddof=1)
print(MAE, RMSE, r2_score, out_mean, out_std)

print('Running time: %s Seconds' % (end-start))

start = time.clock()
y_pred = pipe_SVM.predict([X_test[100]])
end = time.clock()
print('Test time: %s Seconds' % (end-start))


# svr_op = svm.SVR(kernel='rbf')
# # C：错误容忍，越大不能容忍。针对噪声的调节
# # gamma：复杂程度，越大越复杂。rbf核
# params = {'C': [0.01, 0.1, 0.5, 1, 10], 'gamma': [0.01, 0.1, 0.5, 1, 10]}
# grid_search = GridSearchCV(svr_op, param_grid=params, scoring='explained_variance', cv=5)
# grid_search.fit(X, y)
# print("最优化参数组合:", grid_search.best_params_)
