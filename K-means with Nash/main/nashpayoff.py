import math
import numpy as np
import pandas as pd

_DELTA_ = 0.00001


def getStat(rating) :
    ret = []
    for idx, item in rating.T.iterrows() :
        ret.append([item.count(), item.sum()])
    return ret

def deltaMatrix(rating) :
    ret = []
    stat = getStat(rating)
    for uidx, user in rating.iterrows() :
        iidx = 0
        tmp = []
        for item in user :
            userMean = stat[iidx][1] / stat[iidx][0]
            hasRated = not math.isnan(item)
            deltaSum = stat[iidx][1] - (item if hasRated else 0)
            deltaCount = stat[iidx][0] - (1 if hasRated else 0)
            tmp.append((userMean * (deltaCount + _DELTA_)) / (deltaSum  + _DELTA_))
            iidx += 1
        ret.append(tmp)
    return np.array(ret)

def payoff(C, U, rating, delta_train) :
    stat = getStat(rating)
    ret1 = []
    ret2 = []
    delta_UC = []
    for i in range(1,len(stat)) :
        hasCRated = not math.isnan(rating.iloc[C, i])
        hasURated = not math.isnan(rating.iloc[U, i])
        deltaSum = stat[i][0] - (rating.iloc[C, i] if hasCRated else 0) - (rating.iloc[U, i] if hasURated else 0)
        deltaCount = stat[i][1] - (1 if hasCRated else 0) - (1 if hasURated else 0)
        delta_UC.append((deltaSum  + _DELTA_) / (deltaCount + _DELTA_))
    
    for i in range(1,len(stat)) :
        tmp = []
        for j in range(1,len(stat)) :
            tmp.append((delta_train[C][j] if i != j else (stat[i][1] / stat[i][0])) / delta_UC[i-1])
        ret1.append(tmp)
    
    for i in range(1,len(stat)) :
        tmp = []
        for j in range(1,len(stat)) :
            tmp.append((delta_train[U][j] if i != j else (stat[i][1] / stat[i][0])) / delta_UC[i-1])
        ret2.append(tmp)
    
    return np.array(ret1), np.array(ret2)

def pruning(N, payoffMatrix) :
    #IMPORTANT : N strategy of each players, Output will be (2N * 2N)
    df1 = pd.DataFrame(payoffMatrix[0])[0:1].sort_values(by=0, axis=1, ascending=False).iloc[0,:N]
    df2 = pd.DataFrame(payoffMatrix[1])[0:1].sort_values(by=0, axis=1, ascending=False).iloc[0,:N]
    candidate = (df1.append(df2)).sort_index().index
    ret = []
    tmp1 = []
    tmp2 = []
    app1 = []
    app2 = []
    for i in df1.index :
        tmp1.append(payoffMatrix[0][i][:])
        tmp2.append(payoffMatrix[1][i][:])
    for i in df1.index :
        app1.append(np.transpose(tmp1)[i])
        app2.append(np.transpose(tmp2)[i])
        
    ret.append(np.transpose(app1))
    ret.append(np.transpose(app2))
    return ret




