import numpy as np
import torch
from torch import nn

import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

npy = np.load('./Datasets/circRNA-RBP/AGO1/pair.npy',allow_pickle=True)

class wModel(torch.nn.Module):
    def __init__(self):
        super(wModel, self).__init__()

        dims = 7 * 3  
        heads = 3  
        dropout_pro = 0.0  
        self.attentionLayer = torch.nn.MultiheadAttention(embed_dim=dims, num_heads=heads)
        self.x0_linear1 = torch.nn.Linear(84, 21)
        self.x01_linear1 = torch.nn.Linear(21, 21)
        self.linear1 = torch.nn.Linear(21, 1)
        self.linear2 = torch.nn.Linear(101, 2)
        self.conv1dx0 = nn.Conv1d(101, 4, kernel_size=65)
        self.conv1dx1 = nn.Conv1d(99, 4, kernel_size=2)
        self.conv1dx2 = nn.Conv1d(101, 4, kernel_size=81)
        self.outputconv1dx2 = nn.Conv1d(101, 4, kernel_size=1)
        self.BN =nn.BatchNorm1d(4)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.maxpool2d_t_p = nn.MaxPool2d(kernel_size=(1,84), stride=1, padding=0, dilation=1)
        self.maxpool2d_t_end = nn.MaxPool2d(kernel_size=(4,1), stride=1, padding=0, dilation=1)
        self.tsne = TSNE(n_components=2, perplexity=5, n_iter=250)
    def forward(self, data,epoch):
        #data:
        # 0：101,84-kmer
        # 1: 99,21 -KNF
        # 2: 101,101-pair
        Y = data["Y"].to(torch.float32)
        Kmer = data["Kmer"].to(torch.float32)
        x0 = self.x0_linear1(Kmer)
        x1 = data["knf"].to(torch.float32)
        x01=x0+x1

        query = self.x01_linear1(x01)
        key = self.x01_linear1(x01)
        value = self.x01_linear1(x01)
        attn_output, attn_output_weights = self.attentionLayer(query, key, value)

        attn_output = self.outputconv1dx2(attn_output)
        attn_output = self.BN(attn_output)
        attn_output = self.relu(attn_output)

        x2 = data["pair"].to(torch.float32)
        x2 = self.conv1dx2(x2)
        x2_gate = self.sigmoid(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x_gate = x2_gate *attn_output
        x_end = x_gate+x2
        x = self.linear1(x_end)
        x =x.sum(axis=1).squeeze()
        xout = self.sigmoid(x)

        return xout  # 759x1
