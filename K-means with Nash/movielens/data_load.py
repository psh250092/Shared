
import csv
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd

def data_load_100k(file_name: str, verbose):  #입력으로 데이터의 이름이 들어감
    '''
    함수의 입력은 str 타입으로 u1, u2, u3, u4, u5만 받을 수 있다.
    이 이름이 의미하는 것은 ml-100k 폴더에 있는 u1.base, u2.base, u3.base, u4.base, u5.base 파일들이다.
    해당 파일들은 모두 ml-100k의 u.data에서 5-fold validation을 위해 split된 데이터이고, 이대로 사용하면 된다.
    user와 item의 번호는 0부터 시작하는 것으로 한다.

    출력은 train set 을 행렬로 변환시킨 R과 test_set이다.
    '''
    file_path_train = ' '
    file_path_test = ' '
    if file_name == 'u1':
        file_path_train = './ml-100k/u1.base'
        file_path_test = './ml-100k/u1.test'

    elif file_name == 'u2':
        file_path_train = './ml-100k/u2.base'
        file_path_test = './ml-100k/u2.test'
        
    elif file_name == 'u3':
        file_path_train = './ml-100k/u3.base'
        file_path_test = './ml-100k/u3.test'

    elif file_name == 'u4':
        file_path_train = './ml-100k/u4.base'
        file_path_test = './ml-100k/u4.test'

    elif file_name == 'u5':
        file_path_train = './ml-100k/u5.base'
        file_path_test = './ml-100k/u5.test'

    else:
        print('data_load fail !!!!')
        return
    
    user_count = 943
    item_count = 1682

    df = pd.read_csv(file_path_train, sep = ',', names = ['A', 'B', 'C'])
    tmp = df.to_numpy()

    R = np.zeros([user_count, item_count])
    for j in tmp:
        R[j[0]-1][j[1]-1] = j[2]

    df = pd.read_csv(file_path_test, sep = ',', names = ['A', 'B', 'C'])
    test_set = df.to_numpy()

    T = np.zeros([user_count, item_count])
    for j in test_set:
        T[j[0]-1][j[1]-1] = j[2]

    if verbose == True:
        print(file_name, 'data_load success!')
        print('user_count: %d, item_count: %d'%(user_count, item_count))
        
    return R, T


def data_load_1m(file_name: str, verbose):  #입력으로 데이터의 이름이 들어감
    '''
    함수의 입력은 str 타입으로 u1, u2, u3, u4, u5만 받을 수 있다.
    이 이름이 의미하는 것은 ml-100k 폴더에 있는 u1.base, u2.base, u3.base, u4.base, u5.base 파일들이다.
    해당 파일들은 모두 ml-100k의 u.data에서 5-fold validation을 위해 split된 데이터이고, 이대로 사용하면 된다.
    user와 item의 번호는 0부터 시작하는 것으로 한다.

    출력은 train set 을 행렬로 변환시킨 R과 test_set이다.
    '''
    file_path_train = ' '
    file_path_test = ' '
    if file_name == 'u1':
        file_path_train = './ml-1m/u1.base'
        file_path_test = './ml-1m/u1.test'

    elif file_name == 'u2':
        file_path_train = './ml-1m/u2.base'
        file_path_test = './ml-1m/u2.test'
        
    elif file_name == 'u3':
        file_path_train = './ml-1m/u3.base'
        file_path_test = './ml-1m/u3.test'

    elif file_name == 'u4':
        file_path_train = './ml-1m/u4.base'
        file_path_test = './ml-1m/u4.test'

    elif file_name == 'u5':
        file_path_train = './ml-1m/u5.base'
        file_path_test = './ml-1m/u5.test'

    else:
        print('data_load fail !!!!')
        return
    
    user_count = 6040
    item_count = 3952

    df = pd.read_csv(file_path_train, sep = ',', names = ['A', 'B', 'C'])
    #df = df.drop(columns=['D'], axis=1)
    tmp = df.to_numpy()

    R = np.zeros([user_count, item_count])
    for j in tmp:
        R[j[0]-1][j[1]-1] = j[2]

    df = pd.read_csv(file_path_test, sep = ',', names = ['A', 'B', 'C'])
    #df = df.drop(columns=['D'], axis=1)
    test_set = df.to_numpy()
    
    T = np.zeros([user_count, item_count])
    for j in test_set:
        T[j[0]-1][j[1]-1] = j[2]

    if verbose == True:
        print(file_name, 'data_load success!')
        print('user_count: %d, item_count: %d'%(user_count, item_count))
        
    return R, T
