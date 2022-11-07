import pandas as pd
from sklearn.cluster import KMeans
import numpy



# clustering
# @copyright by Sungmin-Ha

# SVD 기반 클러스터링 한 애들을 pure 한 train, test set 에 적용
def clustering(num, SVD_ratings, train_data, test_data):
    
    # 1. clusturing
    km = KMeans(n_clusters=num, init='k-means++')
    cluster = km.fit(SVD_ratings)
    cluster_id = pd.DataFrame(cluster.labels_)                   # 모든  user에 클러스터 n 이 표시됨

    cluster_id.index = SVD_ratings.index                        # userId 칼럼을 인덱스로 설정 1~943
    cluster_id.rename(columns = {0 : 'cluster'}, inplace = True) # 모든  user의 클러스터


    ## 2. train, test 에 cluster 정보 추가
    train_data = pd.concat([train_data, cluster_id], axis=1, join='inner')
    test_data = pd.concat([test_data, cluster_id], axis=1, join='inner')

    km_center = km.cluster_centers_ # 중심
    #print(km.feature_names_in_)
    return train_data, test_data, km_center

# matrix_factorization()
# @copyright by https://github.com/Jaehyung-Lim

def matrix_factorization(R, P, Q, K, steps=300, alpha=0.0002, beta=0.02):

    

    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter
    '''
    Q = Q.T

    start = time.time()
    for step in range(steps):
        print('epoch: %d, time: %f'%(step, time.time()-start))
        for i in range(len(R)):
            for j in range(len(R[0])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = numpy.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[0])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T