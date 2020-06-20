# VDA-KATZ model when k = 2
import pandas as pd
import numpy as np
import math
import random
from sklearn.metrics import roc_auc_score, auc, roc_curve
def getSimilarMatrix(IP, γ_):
    dimensional = IP.shape[0]
    sd = np.zeros(dimensional)
    K = np.zeros((dimensional, dimensional))
    for i in range(dimensional):
        sd[i] = np.linalg.norm(IP[i]) ** 2
    gamad = γ_*dimensional / np.sum(sd.transpose())
    for i in range(dimensional):
        for j in range(dimensional):
            K[i][j] = math.exp(-gamad * (np.linalg.norm(IP[i] - IP[j])) ** 2)
    return K
def Kfoldcrossclassify(sample, K, fun="cv3"):
    r = []
    if fun != "cv3":
        m = np.mat(sample)
        if fun == "cv1":
            t = 0
        else:
            t = 1
        mt = Kfoldcrossclassify(np.array(range(np.max(m[:, t]) + 1)), K)
        r = [[j for j in sample if j[t] in mt[i]] for i in range(K)]
        return r
    l = sample.shape[0]
    t = sample.copy()
    n = math.floor(l / K)
    retain = l - n * K
    for i in range(K - 1):
        nt = n
        e = len(t)
        # if e % n and e % K:
        if retain > i:
            nt += 1
        a = random.sample(range(e), nt)
        r.append([t[i] for i in a])
        t = [t[i] for i in range(e) if (i not in a)]
    r.append(t)
    return r
A = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\association.xls",header=None)
A = A.to_numpy()
Nd, Nv = A.shape
a = [(i, j) for i in range(Nd) for j in range(Nv) if A[i, j]]
a = np.array(a)
b = [(i, j) for i in range(Nd) for j in range(Nv) if A[i, j] == 0]
b = np.array(b)
beta = 0.04
KD = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\small_molecule_drug_sim.xlsx",header=None)
KD = KD.to_numpy()
KV = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\virus_sim(2020.2.12).xlsx",header=None)
KV = KV.to_numpy()
note = []
AUCs, ACCs, SENs, SPEs = (0, 0, 0, 0)
for h in range(100):
    f = Kfoldcrossclassify(a, 5, fun="cv3")
    sum, ACC, SEN, SPE = (0, 0, 0, 0)
    for i in range(5):
        test_sample = np.array(f[i])
        negative_sample = np.array(b)
        A_ = A.copy()
        A_[test_sample[:, 0], test_sample[:, 1]] = 0
        w1 = 0.9
        w2 = 0.9
        GV = getSimilarMatrix(A_.T,2.5)
        GD = getSimilarMatrix(A_,2.5)
        SV = w1 * GV + (1-w1) * KV
        SD = w2 * GD + (1-w2) * KD
        PK2 = beta * A_.T + math.pow(beta, 2) * (SV @ A_.T + A_.T @ SD)
        PK2 = PK2.T
        test_sample_number = test_sample.shape[0]
        negative_sample_number = negative_sample.shape[0]
        label = test_sample_number*[1]+negative_sample_number*[0]
        label = np.array(label)
        sample = np.vstack((test_sample, negative_sample))
        score = PK2[sample[:, 0], sample[:, 1]]
        fpr, tpr, threshold = roc_curve(label,score)
        sumACC = 0
        sumSEN = 0
        sumSPE = 0
        for j in range(threshold.size):
            TP, TN, FP, FN = (0, 0, 0, 0)
            threshold_value = threshold[j]
            for k in range(score.size):
                predicted_value = score[k]
                if predicted_value >= threshold_value:
                    if label[k]:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if label[k]:
                        FN += 1
                    else:
                        TN += 1
            sumACC += (TP+TN)/(TP+TN+FP+FN)
            sumSEN += TP/(TP+FN)
            sumSPE += TN/(TN+FP)
        ACC += sumACC / (threshold.size)
        SEN += sumSEN / (threshold.size)
        SPE += sumSPE / (threshold.size)
        auc_pre = auc(fpr, tpr)
        sum += auc_pre
        note.append((auc_pre, fpr, tpr))
    AUC_mean = sum/5
    ACC_mean = ACC/5
    SEN_mean = SEN/5
    SPE_mean = SPE/5
    AUCs += AUC_mean
    ACCs += ACC_mean
    SENs += SEN_mean
    SPEs += SPE_mean
mm = 0
kk = 0
for x in range(len(note)):
    mm += note[x][0]
nn = mm/len(note)
t_auc = np.inf
dd = None
for y in range(len(note)):
    kk = abs(note[y][0]-nn)
    if kk < t_auc:
        t_auc = kk
        dd = (note[y][1],note[y][2])
fpr, tpr = dd
with open("fpr.txt", "w") as fp:
    for i in fpr:
        fp.write(str(i)+' ')
with open('tpr.txt', 'w') as fp:
    for i in tpr:
        fp.write(str(i)+' ')
AUCs_mean = AUCs/100
ACCs_mean = ACCs/100
SENs_mean = SENs/100
SPEs_mean = SPEs/100
print("the value of AUCs_mean：", AUCs_mean)
print("the value of ACCs_mean：", ACCs_mean)
print("the value of SENs_mean：", SENs_mean)
print("the value of SPEs_mean：", SPEs_mean)
