import pandas as pd
import numpy as np
import random

df = pd.read_csv('secondhand_cost_estimation.csv')
df1 = pd.read_csv('TEST.csv')
nI = 10
nO = 1
nH = int(input('Enter number of hidden neurons :'))
patterns = 1000
patternst = 99
lr = 0.5
tol = 0.01
I = np.zeros((nI, patterns))
O = np.zeros((nO, patterns))
aI = np.transpose(df.values[:, 0:nI])
for i in range(nI):
    I[i] = 0.8 * (aI[i] - min(aI[i])) / ((max(aI[i])) - min(aI[i])) + 0.1
aO = np.transpose(df.values[:, nI:nI + nO])
for i in range(nO):
    O[i] = 0.8 * (aI[i] - min(aI[i])) / ((max(aI[i])) - min(aI[i])) + 0.1
I1 = np.ones((1, patterns))
I = np.vstack([I1, I])
V = np.random.uniform(-1, 1, (nI + 1, nH + 1))
W = np.random.uniform(-1, 1, (nH + 1, nO))
IH = np.zeros((nH + 1, patterns))
OH = np.zeros((nH, patterns))
OH = np.vstack([I1, OH])
IO = np.zeros((nO, patterns))
OO = np.zeros((nO, patterns))
E = np.zeros((nO, patterns))
x = np.array(range(patterns))


def ForwardPass(I, O, V, W, nI, nH, nO, patterns):
    random.shuffle(x)
    IH = np.zeros((nH + 1, patterns))
    IO = np.zeros((nO, patterns))
    for p in x:
        for j in range(1, nH + 1):
            for i in range(nI + 1):
                IH[j][p] = IH[j][p] + I[i][p] * V[i][j]
            OH[j][p] = 1 / (1 + np.exp(-IH[j][p]))
        for k in range(nO):
            for j in range(nH + 1):
                IO[k][p] = IO[k][p] + OH[j][p] * W[j][k]
            OO[k][p] = 1 / (1 + np.exp(-IO[k][p]))
            E[k][p] = 0.5 * (OO[k][p] - O[k][p]) * (OO[k][p] - O[k][p])


def BackwardProp(O, OO, OH, W, V, I, nH, nO, nI, patterns):
    for j in range(nH + 1):
        for k in range(nO):
            for p in range(patterns):
                W[j][k] += (lr / patterns) * ((O[k][p] - OO[k][p]) * OO[k][p] * (1 - OO[k][p]) * OH[j][p])
    for i in range(nI + 1):
        for j in range(1, nH + 1):
            for k in range(nO):
                for p in range(patterns):
                    V[i][j] += (lr / (nO * patterns)) * (
                                (O[k][p] - OO[k][p]) * OO[k][p] * (1 - OO[k][p]) * W[j][k] * OH[j][p] * (1 - OH[j][p]) *
                                I[i][p])


for e in range(10):

    ForwardPass(I, O, V, W, nI, nH, nO, patterns)
    BackwardProp(O, OO, OH, W, V, I, nH, nO, nI, patterns)

    if (np.mean(E[0]) ) <= tol:
        print(e + 1)
        print('Model converges after ' + str(e + 1))
        break

# Training of ANN
Itest = np.zeros((nI, patternst))
Otest = np.zeros((nO, patternst))
aItest = np.transpose(df1.values[:, 0:nI])
for i in range(nI):
    Itest[i] = 0.8 * (aItest[i] - min(aItest[i])) / ((max(aItest[i])) - min(aItest[i])) + 0.1
aOtest = np.transpose(df1.values[:, nI:nI + nO])
for i in range(nO):
    Otest[i] = 0.8 * (aItest[i] - min(aItest[i])) / ((max(aItest[i])) - min(aItest[i])) + 0.1
I1test = np.ones((1, patternst))
Itest = np.vstack([I1test, Itest])
IHt = np.zeros((nH + 1, patternst))
OHt = np.zeros((nH, patternst))
OHt = np.vstack([I1test, OHt])
IOt = np.zeros((nO, patternst))
OOt = np.zeros((nO, patternst))
Et = np.zeros((nO, patternst))
xt = np.array(range(patternst))


def Test(Itest, Otest, V, W, nI, nH, nO, patternst):
    random.shuffle(xt)
    IHt = np.zeros((nH + 1, patternst))
    IOt = np.zeros((nO, patternst))
    for p in xt:
        for j in range(1, nH + 1):
            for i in range(nI + 1):
                IHt[j][p] = IHt[j][p] + Itest[i][p] * V[i][j]
            OHt[j][p] = 1 / (1 + np.exp(-IHt[j][p]))
        for k in range(nO):
            for j in range(nH + 1):
                IOt[k][p] = IOt[k][p] + OHt[j][p] * W[j][k]
            OOt[k][p] = 1 / (1 + np.exp(-IOt[k][p]))
            OOt[k][p] = (OOt[k][p] - 0.1) * ((max(Otest[k])) - min(Otest[k])) / 0.8 + min(Otest[k])
            Et[k][p] = 0.5 * (OOt[k][p] - Otest[k][p]) * (OOt[k][p] - Otest[k][p])
    print('mean error in the output  =  ',np.mean(Et[0]))


Test(Itest, Otest, V, W, nI, nH, nO, patternst)


