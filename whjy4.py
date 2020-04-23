import pandas as pd
import numpy as np
import math
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
A = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\association.xls", header=None)
A = A.to_numpy()
beta = 0.01
KD = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\small_molecule_drug_sim.xlsx", header=None)
KD = KD.to_numpy()
KV = pd.read_excel(r"C:\Users\wjj\VDA-KATZ\data\virus_sim(2020.2.12).xlsx", header=None)
KV = KV.to_numpy()
w1 = 0.9
w2 = 0.9
GV = getSimilarMatrix(A.T, 2.5)
GD = getSimilarMatrix(A, 2.5)
SV = w1 * GV + (1 - w1) * KV
SD = w2 * GD + (1 - w2) * KD
# print(A_)
PK2 = beta * A.T + math.pow(beta,2) * (SV @ A.T + A.T @ SD)   # 算法
PK3 = PK2 + math.pow(beta,3) * (A.T @ A @ A.T + SV @ SV @ A.T + SV @ A.T @ SD + A.T @ SD @ SD)
PK4 = PK3 + math.pow(beta,4) * (SV @ SV @ SV @ A.T + A.T @ A @ SV @ A.T + SV @ A.T @ A @ A.T + A.T @ SD @ A @ A.T)
+ math.pow(beta,4) * (A.T @ A @ A.T @ SD + SV @ SV @ A.T @ SD + SV @ A.T @ SD @ SD + A.T @ SD @ SD @ SD)
PK4 = PK4.T    # 转置a = SK4[:,0]
a = PK4[:,0]
print(a)
b = np.argsort(-a)
b = np.array([b])
c = pd.DataFrame(b.T)
c.to_excel("pk3.xlsx",index=False,header=False)
print(b)
