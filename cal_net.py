import numpy as np
from scipy import sparse
from tqdm import tqdm
import math
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rna_data', type=str, default='./preprocessdata.csv', help='path of rna data file')
parser.add_argument('--outdir', type=str, default="./", help='the directory for output.')
parser.add_argument('--sc', type=int, default=1, help='whether single cell data')
opt = parser.parse_args()

def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

def cal_pccs(x, y,sc):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    if sc:
        i1 = (x>0)+0
        i2 = (y>0)+0
        x = x * i2
        y = y * i1
        x = np.delete(x,np.where(x==0.0))
        y = np.delete(y,np.where(y==0.0))
    n = len(x)
    if n <=0:
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


print("Loading data")
rna = pd.read_csv(opt.rna_data, header=0, index_col=0)
rna = rna.to_numpy()
# adj = np.corrcoef(rna)
# adj = np.around(adj,4)

adj = np.zeros((len(rna),len(rna)))
print(f"Finish loading\n")
for i in tqdm(range(len(rna)-1)):
    if i %100 ==0:
        print(i)
    for j in range(i+1,len(rna)):
        x = rna[i]
        y = rna[j]
        pcc = cal_pccs(x,y,opt.sc)
        if abs(pcc) >=0.3:
            adj[i][j] = np.around(pcc, 4)
            adj[j][i] = np.around(pcc, 4)

allmatrix_sp = sparse.csr_matrix(adj)

if opt.sc:
    sparse.save_npz(opt.outdir+'sc.npz', allmatrix_sp)
else:
    sparse.save_npz(opt.outdir+'bulk.npz', allmatrix_sp)
