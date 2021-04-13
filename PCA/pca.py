import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataprocess import get_normalize


def mean_of_matrix(X):
    a = X.shape[0]
    tempX = 0
    for i in range(a):
        tempX += X[i]
    if len(tempX.shape) == 1:
        tempX = tempX.reshape((1, -1))
    return tempX / a


def var_matrix(data):
    """
    这里的data是已经中心化之后的data了
    :param data: 输入均值化的数据
    :return: 协方差矩阵
    """
    m, n = data.shape
    S = data.T @ data
    S /= m
    return S


def pca(data):
    print('Using PCA...')
    S = var_matrix(data)
    eig_value, vec = np.linalg.eigh(S)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    vec = vec.T
    W = [vec[index[i]] for i in range(len(index))]
    return np.array(W)


def pcaProject(data, dimension=120):
    data = get_normalize(data)
    project_W = pca(data)
    project_W = project_W[:dimension]
    return data @ project_W.T
