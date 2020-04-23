import random

import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score
from random import shuffle

def getSimilarMatrix(IP, γ_):
    dimensional = IP.shape[0]
    sd = np.zeros(dimensional)
    K = np.zeros((dimensional, dimensional))
    for i in range(dimensional):
        sd[i] = np.linalg.norm(IP[i]) ** 2
    gamad =  γ_*dimensional / np.sum(sd.transpose())
    for i in range(dimensional):
        for j in range(dimensional):
            K[i][j] = math.exp(-gamad * (np.linalg.norm(IP[i] - IP[j])) ** 2)
    return K

A = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\association.xls", header=None)
A = A.to_numpy()
# rows = np.where(A == 1)[0]
# cols = np.where(A == 1)[1]
# # print(cols)
# state = np.random.get_state()
# np.random.shuffle(rows)
# np.random.set_state(state)
# np.random.shuffle(cols)
beta = 0.01
# print(rows)
# print(cols)
KD = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\small_molecule_drug_sim.xlsx",header=None)
KD = KD.to_numpy()
KV = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\virus_sim(2020.2.12).xlsx",header=None)
KV = KV.to_numpy()
w1 = 0.9
w2 = 0.9
GV = getSimilarMatrix(A.T,2.5)
GD = getSimilarMatrix(A,2.5)
SV = w1 * GV + (1-w1) * KV
SD = w2 * GD + (1-w2) * KD
PK2 = beta *A.T + math.pow(beta, 2) * (SV @ A.T +A.T @ SD)
PK2 = PK2.T
a = PK2[:,0]
print(a)
b = np.argsort(-a)
b = np.array([b])
c = pd.DataFrame(b.T)
c.to_excel("pk1.xlsx",index=False,header=False)
print(b)
