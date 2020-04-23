import pandas as pd
import numpy as np
import math
import random
from sklearn.metrics import roc_auc_score, auc,roc_curve
from matplotlib import pyplot as plt
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
        # for i in range(K):
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
# print(A)
# 将表格转换成矩阵
A = A.to_numpy()
# 把矩阵A的行列数值赋给变量
Nd, Nv = A.shape
# 取=1的位置
a = [(i, j) for i in range(Nd) for j in range(Nv) if A[i, j]]
a = np.array(a)
# 取=0的位置
b = [(i, j) for i in range(Nd) for j in range(Nv) if A[i, j] == 0]
b = np.array(b)

beta = 0.01
KD = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\small_molecule_drug_sim.xlsx",header=None)
KD = KD.to_numpy()
# print(KD)
KV = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\virus_sim(2020.2.12).xlsx",header=None)
KV = KV.to_numpy()
# print(KV)
note = []
AUCs, ACCs, SENs, SPEs = (0, 0, 0, 0)
for h in range(100):

    f = Kfoldcrossclassify(a, 5, fun="cv3")
    sum, ACC, SEN, SPE = (0, 0, 0, 0)
    for i in range(5):

        test_sample = np.array(f[i])     #  多行两列矩阵存的是一的坐标
        negative_sample = np.array(b)
        # 变为矩阵
        A_ = A.copy()
        A_[test_sample[:, 0], test_sample[:, 1]] = 0  # 取测试样本的行和列   注意text_sample用法  20或19个1已经换成0了
        w1 = 0.9
        w2 = 0.9
        GV = getSimilarMatrix(A_.T, 2.5)
        GD = getSimilarMatrix(A_, 2.5)
        SV = w1 * GV + (1 - w1) * KV
        SD = w2 * GD + (1 - w2) * KD
        # print(A_)
        PK2 = beta * A_.T + math.pow(beta,2) * (SV @ A_.T + A_.T @ SD)   # 算法
        PK3 = PK2 + math.pow(beta,3) * (A_.T @ A_ @ A_.T + SV @ SV @ A_.T + SV @ A_.T @ SD + A_.T @ SD @ SD)
        PK3 = PK3.T    # 转置
        test_sample_number = test_sample.shape[0]    # 去测试样本的行
        # print(test_sample_number)
        negative_sample_number = negative_sample.shape[0]  # 取负样本的行
        # print(negative_sample_number)
        label = test_sample_number*[1]+negative_sample_number*[0] # 变成【。，。，。，。】为列表一行860/859列
        # print(label)
        label = np.array(label)   # 变为矩阵  一行
        # print(label)
        sample = np.vstack((test_sample,negative_sample)) # 变为多行 两列的矩阵  上面为1的位置下边为0的位置
        score = PK3[sample[:, 0],sample[:, 1]] # 因为之前已经有19或20个数变了，所以取得预测值也就变了，目的是取预测值，一行860/859列
        # print(score.shape)
        fpr,tpr,threshold = roc_curve(label,score)   # 求AUC 必写1

        sumACC = 0                 # 下面为求ACC的方法
        sumSEN = 0
        sumSPE = 0
        for j in range(threshold.size):
            TP,TN,FP,FN=(0,0,0,0)
            yuzhi = threshold[j]
            for k in range(score.size):
                yuce = score[k]
                if yuce > yuzhi:
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

        ACC += sumACC/(threshold.size)  # 求acc的平均
        SEN += sumSEN/(threshold.size)
        SPE += sumSPE/(threshold.size)

        # plt.plot(fpr,tpr)  # 折线图  fpr为横坐标   tpr 为纵坐标
        # plt.show()      # 让电脑画图
        auc_pre = auc(fpr,tpr)    # 求AUC 必写2
        sum += auc_pre # aucp 求和
        note.append((auc_pre, fpr, tpr))
        # print("auc_pre的值是：",auc_pre)
        # print("ACC的值是：",ACC)
        # print("SEN的值是：",SEN)
        # print("SPE的值是：",SPE)
    AUC_mean = sum/5
    ACC_mean = ACC/5
    SEN_mean = SEN/5
    SPE_mean = SPE/5
    # print("AUC_mean的值是：",AUC_mean)    # 取平均值
    # print("ACC_mean的值是：",ACC_mean)
    # print("SEN_mean的值是：",SEN_mean)
    # print("SPE_mean的值是：",SPE_mean)
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
fpr,tpr = dd
with open("fprk3.txt","w") as fp:
    for i in fpr:
        fp.write(str(i)+' ')
with open("tprk3.txt",'w') as fp:
    for i in tpr:
        fp.write(str(i)+' ')
# plt.plot(fpr,tpr)  # 折线图  fpr为横坐标   tpr 为纵坐标
# plt.show()
AUCs_mean = AUCs/100
ACCs_mean = ACCs/100
SENs_mean = SENs/100
SPEs_mean = SPEs/100
print("AUCs_mean的值是：", AUCs_mean)
print("ACCs_mean的值是：", ACCs_mean)
print("SENs_mean的值是：", SENs_mean)
print("SPEs_mean的值是：", SPEs_mean)




