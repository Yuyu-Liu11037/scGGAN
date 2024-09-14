from __future__ import print_function, division
import argparse
import os
import numpy as np
import pandas as pd
import math
import sys
from tqdm import tqdm
import torchvision.transforms as transforms
from torch import cuda
import time
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import random
from numpy import unique
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,calinski_harabasz_score,silhouette_score,f1_score,precision_score,recall_score
from umap import UMAP
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
import matplotlib.pyplot as plt
import models
from load_data import load_RG,load_data,loadexp,laplacian,r_norm,normalized_laplacian,unnormalized_laplacian
from scipy import sparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=50, help='size of the batches')
parser.add_argument('--kt', type=float, default=0, help='kt parameters')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma parameters')
parser.add_argument('--lr', type=float, default=0.01, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--graph', type=int, default=1, help='graph weight')
parser.add_argument('--gpu', type=int, default=0, help='cuda gpuid')
parser.add_argument('--dpt', type=str, default='', help='load discrimnator model')
parser.add_argument('--gpt', type=str, default='', help='load generator model')
parser.add_argument('--train', help='train the network', action='store_true')
parser.add_argument('--seed', type=int,default=666,help='random seed')
parser.add_argument('--knn_k', type=int, default=10, help='neighours used')
parser.add_argument('--rna_data', type=str, default='./preprocessdata.csv', help='path of rna data file')  # cell*gene
parser.add_argument('--label', type=str, default='./label.csv', help='path of label file')
parser.add_argument('--bulk_net', type=str, default='./bulk.npz', help='path of bulk network file')
parser.add_argument('--bulk', type=int, default=0, help='')
parser.add_argument('--sc_net', type=str, default='./sc.npz', help='path of single cell network file')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
parser.add_argument('--cl', type=int, default=2, help='number of label')
parser.add_argument('--gene_data', type=str, default='./4mergenevec.csv', help='path of gene data file')  # gene pair
parser.add_argument('--lr_rate', type=int, default=10, help='rate for slow learning')
parser.add_argument('--outdir', type=str, default="./", help='the directory for output.')
opt = parser.parse_args()
seed= opt.seed
np.random.seed(seed)
torch.manual_seed(seed)

AE_models = opt.outdir + 'GANs_models/'
if not os.path.exists(AE_models):
	os.makedirs(AE_models)

if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(seed)
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda.set_device(opt.gpu)
    print('Using GPU ' + str(opt.gpu))
else:
    dtype = torch.FloatTensor

G = models.G(350, 256, 256, 128)
D = models.D(350, 256, 256, 128)

G.type(dtype)
D.type(dtype)
print("model have {} paramerters in total".format(sum(x.numel() for x in G.parameters())))
for p in G.parameters():
    p.data.uniform_(-0.1, 0.1)
for p in D.parameters():
    p.data.uniform_(-0.1, 0.1)
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr)


class loss_tri(nn.Module):
    def __init__(self):
        super(loss_tri, self).__init__()
        self.s1 = nn.Parameter(torch.ones(1,requires_grad=True))
        self.s2 = nn.Parameter(torch.ones(1,requires_grad=True))

    def forward(self,l1,l2):
        loss = 1/(2*self.s1**2) *l1 + 1/(2*self.s2**2) *l2+ torch.log(self.s1 *self.s2)
        return loss

class loss_dis(nn.Module):
    def __init__(self):
        super(loss_dis, self).__init__()

    def forward(self,x,y,m):
        loss = abs(x-y)
        loss = torch.mean(loss[m])
        return loss

def train_AE(rna,gene,L):

    print("model training")
    loss_fn = loss_dis().type(dtype)
    loss_a = loss_tri().type(dtype)

    d_loss_hist = []
    g_loss_hist = []
    g_tloss_hist = []
    dis_loss_hist = []
    min_dis_loss = 1

    for epoch in range(opt.n_epochs):
        G.train()
        D.train()
        D.zero_grad()

        # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)
        D_total_loss = 0
        G_t_loss = 0
        G_total_loss = 0
        dis_total_loss = 0

        mask = (rna > 0) + 0
        mask = mask.detach()
        data = rna
        mask = mask.type(dtype)

        rna_i = G(data, gene, L)
        rna_i = rna_i.detach()
        hat = rna * mask + rna_i * (1 - mask)
        p = D(hat,gene, L)
        c_loss = -torch.mean(mask * torch.log(p + 1e-8)) -torch.mean((1 - mask) * torch.log(1. - p + 1e-8))
        c_loss.backward()
        optimizer_D.step()
        D_total_loss += c_loss.item()

        G.zero_grad()

        rna_i = G(data, gene, L)
        hat = rna * mask + rna_i * (1 - mask)
        p = D(hat,gene, L)

        G_loss_temp = -torch.mean((1 - mask) * torch.log(p + 1e-8))
        a = torch.BoolTensor(mask.cpu().bool())
        dis_loss = loss_fn(rna, rna_i,a)
        G_loss = loss_a(G_loss_temp,dis_loss)
        G_loss.backward()
        optimizer_G.step()
        G_t_loss += G_loss.item()

        G_total_loss += G_loss_temp.item()
        dis_total_loss += dis_loss.item()

        print('Epoch {}: Train D loss: {:.4f}, G loss: {:.4f}, LG: {:.4f}, LDis: {:.4f}'.format(epoch,D_total_loss,G_t_loss, G_total_loss,dis_total_loss))
        d_loss_hist.append(D_total_loss)
        g_loss_hist.append(G_total_loss)
        g_tloss_hist.append(G_t_loss)
        dis_loss_hist.append(dis_total_loss)
        if dis_loss<min_dis_loss:
            min_dis_loss = dis_loss
            torch.save(G, "./GANs_models/generator.pt")
            torch.save(D, "./GANs_models/discriminator.pt")

    return d_loss_hist,g_tloss_hist,g_loss_hist,dis_loss_hist


def impute(rna,gene,L):
    mask = (rna > 0) + 0
    data = rna
    G = torch.load("./GANs_models/generator.pt")
    G.eval()

    rna_i = G(data, gene, L)

    rna_i = rna * mask + rna_i * (1 - mask)

    return rna_i.cpu().detach().numpy()

rna_file = opt.rna_data
gene_file = opt.gene_data

rna,gene,y = load_RG(opt.rna_data,opt.gene_data,opt.batch_size)
gene = gene.type(dtype)
rna = rna.type(dtype)
if opt.bulk:
    allmatrix_sp = sparse.load_npz(opt.bulk_net)
    adj1 = allmatrix_sp.toarray()
    allmatrix_sp = sparse.load_npz(opt.sc_net)
    adj2 = allmatrix_sp.toarray()
    adj = (opt.alpha * adj1 + adj2)/(opt.alpha + 1)
else:
    allmatrix_sp = sparse.load_npz(opt.sc_net)
    adj = allmatrix_sp.toarray()


L = normalized_laplacian((adj+1)/2)
L = torch.from_numpy(L).type(dtype)

label = np.loadtxt(opt.label)
d_loss,g_tloss,g_loss,dis = train_AE(rna,gene,L)
impute_data = impute(rna,gene,L)
# impute_data = impute_data * y
tsne = UMAP()
result = tsne.fit_transform(np.transpose(impute_data))

impute_data = np.around(impute_data).astype(int)
pd.DataFrame(impute_data).to_csv("./data/impute.csv",header=False,index=False)


# print('Computing T-sne embedding')

# c = KMeans(n_clusters=opt.cl)
# c.fit(result)
# y = c.predict(result)
# ar = adjusted_rand_score(label,y)
# nmi = normalized_mutual_info_score(label,y)
# np.savetxt('./label.txt',y)
# plt.scatter(result[:, 0], result[:, 1], c=label)
# print(ar,nmi)
# plt.show()