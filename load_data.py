from __future__ import print_function, division
import argparse
import os
import numpy as np
import pandas as pd
import math
import sys
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from scipy import sparse

def loadexp(csv_file):
    data = pd.read_csv(csv_file, header=0, index_col=0)
    gene = data.index.tolist()
    data = data.to_numpy().astype(np.float)
    data = np.transpose(data)
    # data = np.log2(data+1)
    data,y = normalize(data)
    return data,gene,y

def loadgene(csv_file):
    gene = pd.read_csv(csv_file,header=None,index_col=0)
    gene = gene.to_numpy().astype(np.float)
    return gene

def normalize(X):
    print("begin normalize")
    y = []
    for i in range(X.shape[0]):
        max = np.max(X[i])
        y.append(max)
        if max>0:
            X[i] = X[i]/float(max)
    return X,y

def r_norm(X,y):
    return X*y


'''
@input
filename: path to network in in gene pairs format (gene1","gene2","edge weight\n)
genename: csv gene 
@output
net
'''
def import_network_from_gene_pairs(filename,genename):
    print("load GGI network")
    with open(filename) as f:
        genepairs = f.read().split('\n')
    genepairs = [genes.split(',') for genes in genepairs]
    # genes = list(set([gene for genes in genepairs for gene in genes if not gene.replace('.','',1).isdigit()]))
    # genes = [gene for gene in genes if len(gene) > 2]
    network = np.zeros([len(genename),len(genename)])
    i = 0
    for g in tqdm(genepairs):
        if len(g) > 1:
            gene1 = g[0].upper()
            gene2 = g[1].upper()
            if len(g)>2 and g[2].replace('.','',1).isdigit():
                num = float(g[2])
            else:
                num = 1
            try:
                ind1 = genename.index(gene1)
                ind2 = genename.index(gene2)
                network[ind1,ind2] = num
                network[ind2,ind1] = num
                i += 1
            except:
                continue
    print("Total %d edges" %i)
    return network

class GRDataset(Dataset):
    def __init__(self,rna,gene):
        self.rna = rna
        self.gene = gene

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        rna = self.rna[idx]
        rna = torch.unsqueeze(rna,dim=-1)
        rna = torch.cat((rna,self.gene),dim=-1)

        # sample = {"rna":rna}
        return rna

class MyDataset(Dataset):
    """Operations with the datasets."""

    def __init__(self, data, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data[idx]
        data = torch.from_numpy(data)
        data = torch.unsqueeze(data,-1)
        # sample = {'data': data}
        # if self.transform:
        #     sample = self.transform(sample)
        return data
def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def cal_pccs(x, y):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    i1 = (x>0)+0
    i2 = (y>0)+0
    x = x * i2
    y = y* i1
    x = np.delete(x,np.where(x==0.0))
    y = np.delete(y,np.where(y==0.0))
    n = len(x)
    if n <=10:
        return 0
    sum1 = sum(x)
    sum2 = sum(y)

    sumofxy = multipl(x, y)

    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    den = np.sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    pcc = num / den

    # sum_xy = np.sum(np.sum(x*y))
    # sum_x = np.sum(np.sum(x))
    # sum_y = np.sum(np.sum(y))
    # sum_x2 = np.sum(np.sum(x*x))
    # sum_y2 = np.sum(np.sum(y*y))
    # pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))



    if math.isnan(pcc):
        return 0
    return round(pcc,4)
def load_RG(rna_file,gene_file,batch):
    rna, gene_name, y = loadexp(rna_file)
    rna = np.transpose(rna)
    gene = loadgene(gene_file)
    # for i in range(len(rna)-1):
    #     if i %100 ==0:
    #         print(i)
    #     for j in range(i+1,len(rna)):
    #         x = rna[i]
    #         y = rna[j]
    #         pcc = cal_pccs(x,y)
    #         if pcc >=0.4:
    #             adj[i][j] = np.around(pcc, 4)
    #
    # allmatrix_sp = sparse.csr_matrix(adj)
    # sparse.save_npz('allmatrix_sparse.npz', allmatrix_sp)
    # # #
    # allmatrix_sp = sparse.load_npz('time.npz')
    # adj = allmatrix_sp.toarray()


    rna = torch.from_numpy(rna)
    gene = torch.from_numpy(gene)
    # rna = rna.t()
    # adj = torch.from_numpy(adj)
    # adj = torch.corrcoef(rna)
    # train_inputs = GRDataset(rna,gene)

    # Train = torch.utils.data.DataLoader(train_inputs,batch_size=batch, shuffle=True)




    return rna,gene,y



def load_data(csv_file,batch_size):
    '''
    Loads data into a 3D tensor for each of the 3 splits.

    '''
    print("==>loading train data")
    print("load expression data")
    data,gene,y = loadexp(csv_file)
    print("total %d genes" % len(gene))
    train_inputs = MyDataset(data)

    Train = torch.utils.data.DataLoader(train_inputs,batch_size=batch_size, shuffle=True)

    return Train,y

'''
@input 
net: gene x gene numpy array representing a network

@output:
L: laplacian of network
'''
def laplacian(net):
    print("calculate laplacian matrix")
    D = np.sum(net,axis=0) * np.eye(net.shape[0])
    Dpow = np.sqrt(D)
    Dpow = Dpow + 1e-10
    Dpow = 1 / Dpow
    Dpow[Dpow >= 1e10] = 0
    N = np.dot(np.dot(Dpow,net),Dpow)
    L = np.eye(net.shape[0]) - N
    # L = D - net
    return L


def unnormalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adj_matrix



def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / (np.sqrt(R)+ 1e-10)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

