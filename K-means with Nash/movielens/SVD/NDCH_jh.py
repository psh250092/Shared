# @copyright by https://github.com/Jaehyung-Lim
import numpy as np
from math import log2

'''
rel 값은 0,1,2,3,4,5로 둔다. (점수를 그대로 rel 값으로 )
'''



def NDCG(RS, test, k, y, numItems, Nan_to_zero):   #개인에 대한 모든 NDCG 값을 구해야 됨., 각 개인에 대해서 어떤 그룹에 속하는지에 대한 정보도 알아야 됨
    '''
    RS: 그룹 스코어 
    test: test dataset -> Raw Data 형태로 들어옴
    k: 몇개 추천하는 NDCG인지 
    y: 각 개인이 어느 그룹에 속했는지에 대한 정보를 알려줌
    '''
    Nan_to_val = 0

    if Nan_to_zero == False:
        Nan_to_val = 3.52986

    numClusters = len(RS)
    numUsers  = len(y)

    T_ = [[Nan_to_val for j in range(numItems)] for i in range(numUsers)]

    for r in test:
        T_[r[0]-1][r[1]-1] = r[2]

    #이제 test data행렬인 T_도 만들었음.

    Dcg = [0 for i in range(numUsers)]
    Idcg = [0 for i in range(numUsers)]
    Ndcg = [0 for i in range(numUsers)]


    '''DCG step'''
    for i in range(numUsers):  #각각의 사용자에 대해서 판단해야됨
        tmp = RS[i] #User i가 속하는 그룹의 추천 원소
        #tmp[j]: j번째 추천 원소 
        for j, val in enumerate(tmp):
            Dcg[i] += T_[i][val]/log2(j+2)
            
            #rel은 그냥 점수로 했는데, 값 조정하면 됨
            #그냥 User i가 만약 tmp[j]아이템에 대하여 점수가 있을 시에 저렇게 더해줌
             
    '''IDCG step'''
    for i in range(numUsers):
        TMP = [] + T_[i]
        TMP.sort(reverse = True)  # 제일 큰 수대로 값이 나옴 

        for j in range(k):
            Idcg[i] += TMP[j]/log2(j+2)
    

    for i in range(numUsers):
        if Idcg[i] != 0:
            Ndcg[i] = Dcg[i]/Idcg[i]
            
        else:  # 추천 받은게 실제로 없을 때 
            Ndcg[i] = 0

    avg = 0

    for i in Ndcg:
        avg += i

    return avg/numUsers