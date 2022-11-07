import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ndcg_score



def groupRS_ndcg(num, total_matrix, train_matrix, test_matrix):
    '''
    total_matrix #모든 데이터
    train_matrix #트레인 데이터
    test_matrix  #테스트 데이터
    '''
    
    ## 1. K-means로 그룹 clustering(total 대상) 

    km = KMeans(n_clusters=num, init='k-means++')
    cluster = km.fit(total_matrix)
    cluster_id = pd.DataFrame(cluster.labels_)                   # 모든  user에 클러스터 n 이 표시됨

    cluster_id.index = total_matrix.index                        # userId 칼럼을 인덱스로 설정 1~943
    cluster_id.rename(columns = {0 : 'cluster'}, inplace = True) # 모든  user의 클러스터
    
    length = [0]*num
    for i in range(num):
        length[i] = len(cluster_id[cluster_id.cluster==i])       # 각 클러스터에 해당하는 개수
    
    #print("각 cluster 별 user 회원 수 : ", length)


    ## 2. train, test 에 cluster 정보 추가
    user_item_train_cl = pd.concat([train_matrix, cluster_id], axis=1, join='inner')
    user_item_test_cl = pd.concat([test_matrix, cluster_id], axis=1, join='inner')
    

    ## 3. train 대상 - 클러스터 별로 각 item 별 mean 값 구함  < 추천할 아이템의 평점 예측
    mean_rating = pd.DataFrame(columns = user_item_train_cl.columns)
    mean_rating.set_index('cluster') 
    
    # 클러스터별 각 영화 별 별점 평균 계산 (이걸 통해 평균평점이 가장 높은 영화부터 추천)
    for i in range(num):
        mean_rating = mean_rating.append(user_item_train_cl[user_item_train_cl.cluster == i].mean(axis=0), ignore_index=True) 
    
    mean_rating = mean_rating.set_index('cluster')
    #print(mean_rating)


    ## 3. train-test set의 columns(item id) 맞추기 (miss matching 제거) (train, test에 없는 데이터 존재 가능)
    for c in user_item_train_cl.columns:
        if c not in user_item_test_cl.columns:
            del mean_rating[c]
        
    for c in user_item_test_cl.columns:
        if c not in user_item_train_cl.columns:
            del user_item_test_cl[c]
   
    y_pred = mean_rating
    y_true = user_item_test_cl
    
    
    result = [0]*num # 결과값 저장 리스트
    
    #centroids  = km.cluster_centers_
    #print(centroids.shape)
    #labels = km.predict(total_matrix)
    
    #각 클러스터 별 중심점
    #centroid_labels = [centroids[i] for i in labels]
    #print(centroid_labels)

    # nDCG / precision , recall / Average / Least mesary / Average without LM 도출해야함
    
    ## 4. 각 결과 값에 nDCG 더해줌
    for idx in test_matrix.index:

        # -1 해당 user 의 그룹을 확인
        cluster_num = int(y_true.loc[idx].cluster) 
        
        # -2 train 에서의 예측치와 test 데이터 간 각 user 별 평점 NDCG 값 계산
        result[cluster_num] += ndcg_score([y_true.loc[idx][:-1]], [y_pred.loc[cluster_num]])

        # ndcg 는 상위 k의 것을 대상으로 점수를 매길 수 있다.
        #result[cluster] += ndcg_score([user_item_test_cl.loc[idx][:-1]], [mean_rating.loc[cluster]], k=4)
    
    #cl
    
    ## 5. 최종적으로 각 nDCG값 / 각 cluster의 요소 개수
    for i in range(num):
        # 클러스터 별 유저 NDCG 의 평균
        result[i] = result[i]/length[i]
    
    print(f"cluster 수 : {len(length)}\ncluster 별 인원 수 : {length}")
    print(f"총 NDCG : {(sum(result)/len(length)):.4f} \n\n ")