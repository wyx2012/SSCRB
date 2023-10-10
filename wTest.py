import sys
from datetime import time
import pynvml
import numpy as np
import torch.nn as nn

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score
import shap

import wyxPreData
from WOwnDataset import WOwnDataset
from wModle import wModel

from torch.utils.data import Dataset, DataLoader, random_split
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
# %%超参数
protein = "AGO3"
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
NUM_EPOCHS = 50
LR = 0.0002
LOG_INTERVAL = 20
modeling = wModel
cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(0))
print('cuda_name:', cuda_name)


# %%方法
def cal_auc(true_arr, pred_arr):
    # 首先按照概率从高到低排序
    list_ori = []
    for i in range(len(true_arr)):
        list_ori.append((true_arr[i], pred_arr[i]))
    new_list = sorted(list_ori, key=lambda x: x[1], reverse=True)

    auc1 = roc_auc_score(true_arr, pred_arr)
    y_pred = np.around(pred_arr, 0).astype(int)
    acc = accuracy_score(true_arr, y_pred)
    F1 = f1_score(true_arr, y_pred)
    mat = confusion_matrix(true_arr, y_pred)
    tp = float(mat[0][0])
    fp = float(mat[1][0])
    fn = float(mat[0][1])
    tn = float(mat[1][1])
    if (tp + fp) == 0:
        Pre = 1
    else:
        Pre = tp / (tp + fp)
    if (tp + fn) == 0:
        Recall = 1
    else:
        Recall = se = tp / (tp + fn)
    fpr, tpr, thresholds = roc_curve(true_arr, pred_arr)
    auc2 = auc(fpr, tpr)
    # 正样本个数M，负样本个数N
    M, N = 0, 0
    # Z代表，负样本前有多少个正样本，和正样本之后有多少个负样本
    Z = 0
    for item in new_list:
        # 如果是负样本的话
        if item[0] == 0:
            N += 1
            Z += M
        elif item[0] == 1:
            M += 1
    auc0 = Z / (M * N)
    return auc0, auc1, auc2, acc, Pre, Recall, F1


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('model.train()结束')
    for batch_idx, data in enumerate(train_loader):
        data["Kmer"] = data["Kmer"].to(device)
        data["knf"] = data["knf"].to(device)
        data["pair"] = data["pair"].to(device)
        optimizer.zero_grad()
        output = model(data, epoch)
        total_preds = torch.cat((total_preds, output.cpu()), 0)
        total_labels = torch.cat((total_labels, data["Y"].view(-1, 1).cpu()), 0)
        loss = loss_fn(output, data["Y"].float().to(device))
        loss.backward()
        optimizer.step()
        return total_labels.detach().numpy().flatten(), total_preds.detach().numpy().flatten()
        # if batch_idx % LOG_INTERVAL == 0:
        #     print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
        #                                                                    batch_idx * len(data["Kmer"]),
        #                                                                    len(train_loader.dataset),
        #                                                                    100. * batch_idx / len(train_loader),
        #                                                                loss.item()))


def predicting(model, device, loader, epoch):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data["Kmer"] = data["Kmer"].to(device)
            data["knf"] = data["knf"].to(device)
            data["pair"] = data["pair"].to(device)
            output = model(data, epoch)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data["Y"].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


# %%数据准备
if __name__ == '__main__':
    n_train = len(WOwnDataset(protein))
    split = n_train // 5
    auclist0 = []
    auclist1 = []
    auclist2 = []
    acclist = []
    Prelist = []
    Recalllist = []
    F1list = []
    for i in range(1):

        indices = np.random.choice(range(n_train), size=n_train, replace=False)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        train_loader = DataLoader(WOwnDataset(protein), sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
        test_loader = DataLoader(WOwnDataset(protein), sampler=test_sampler, batch_size=TEST_BATCH_SIZE)
        print("train_loader结束")
        # %%训练模型
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        model_file_name = 'w_model.model'


        for epoch in range(NUM_EPOCHS):
            GT, GP = train(model, device, train_loader, optimizer, epoch + 1)
            #  aucT,accT,PreT,RecallT,F1T = cal_auc(GT, GP)
            G, P = predicting(model, device, test_loader, epoch + 1)
            auc0, auc1, auc2, acc, Pre, Recall, F1 = cal_auc(G, P)
            # aucTlist.append(aucT)
            auclist0.append(auc0)
            auclist1.append(auc1)
            auclist2.append(auc2)
            acclist.append(acc)
            Prelist.append(Pre)
            Recalllist.append(Recall)
            F1list.append(F1)
    # with open('results/cTest.txt', 'a') as f:
    #     f.write('train ' + str(aucTlist) + '\n')
    with open('./Datasets/circRNA-RBP/' + protein + '/cTest.txt', 'a') as f:
        f.write('auc0 ' + str(auclist0) + '\n')


        # torch.save(model.state_dict(), model_file_name)
