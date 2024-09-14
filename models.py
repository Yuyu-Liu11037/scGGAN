import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple
import sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import autograd



class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        L*X*\theta
        Args:
        -------
            input_dim: int
            output_dim: int
            use_bias: bool, optional
        """
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # self.net =nn.Linear(input_dim,output_dim,bias=False)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.matmul(input_feature, self.weight)
        # support = self.net(input_feature)
        output = torch.matmul(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

class G(nn.Module):
    def __init__(self,rna_dim,gene_dim,mid_dim,second_dim):
        super(G, self).__init__()
        self.rna_dim = rna_dim
        self.gene_dim = gene_dim
        self.second_dim = second_dim
        self.mid_dim = mid_dim

        self.dropout = nn.Dropout(0.5)

        self.enrna = nn.Sequential(
            nn.Linear(self.rna_dim, 2 * self.mid_dim),
            # nn.BatchNorm1d(2 * self.mid_dim),
            nn.ReLU(),
            nn.Linear(2 * self.mid_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
        )
        self.engene = nn.Sequential(
            nn.Linear(self.gene_dim, 2 * self.mid_dim),
            # nn.BatchNorm1d(2 * self.mid_dim),
            nn.ReLU(),
            nn.Linear(2 * self.mid_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
        )
        self.en1 = GCN(2*self.mid_dim,self.second_dim)
        self.bn = nn.BatchNorm1d(self.second_dim)
        self.en2 = GCN(self.second_dim,self.rna_dim)

        self.en3 = nn.Sequential(
            nn.Linear(self.rna_dim,self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim,self.rna_dim),
        )


    def forward(self,rna,gene,adj):
        rna = self.dropout(rna)
        enrna = self.enrna(rna)
        engene = self.engene(gene)
        data = torch.cat((enrna,engene),dim=-1)
        data = torch.relu((self.en1(adj,data)))
        data = self.en2(adj,data)
        data = torch.relu(data)
        data = self.en3(data)
        data = torch.sigmoid(data)

        return data

class D(nn.Module):
    def __init__(self,rna_dim,gene_dim,mid_dim,second_dim):
        super(D, self).__init__()
        self.rna_dim = rna_dim
        self.gene_dim = gene_dim
        self.second_dim = second_dim
        self.mid_dim = mid_dim

        self.enrna = nn.Sequential(
            nn.Linear(self.rna_dim, 2 * self.mid_dim),
            # nn.BatchNorm1d(2 * self.mid_dim),
            nn.ReLU(),
            nn.Linear(2 * self.mid_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
        )
        self.engene = nn.Sequential(
            nn.Linear(self.gene_dim, 2 * self.mid_dim),
            # nn.BatchNorm1d(2 * self.mid_dim),
            nn.ReLU(),
            nn.Linear(2 * self.mid_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
        )
        self.en1 = GCN(2 * self.mid_dim, self.second_dim)
        self.bn = nn.BatchNorm1d(self.second_dim)
        self.en2 = GCN(self.second_dim, self.rna_dim)

        self.en3 = nn.Sequential(
            nn.Linear(self.rna_dim, self.mid_dim),
            # nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.rna_dim),
        )



    def forward(self,rna,gene,adj):
        enrna = self.enrna(rna)
        engene = self.engene(gene)
        data = torch.cat((enrna,engene),dim=-1)
        data = torch.relu((self.en1(adj,data)))
        data = self.en2(adj,data)
        data = torch.relu(data)
        data = self.en3(data)
        data = torch.sigmoid(data)

        return data