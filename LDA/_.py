import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LDA.creatData import creatData
from LDA.draw import drawFigure


# 按列计算均值, 返回均值向量
def mean_of_matrix(data):
    return np.mean(data, axis=0)


# 计算类内离散度
def compute_Sw(data):
    Sw = 0
    u = mean_of_matrix(data)
    # print("u:")
    # print(u)

    for i in range(0, data.shape[0]):
        print('data[i]:')
        print(data[i])
        Sw = Sw + (data[i] - u).T * (data[i] - u)
        print('data[i, :]:')
        print(data[i, :])
    return Sw


# 计算类间离散度
def compute_Sb(data_set):
    data_all = data_set[0]
    for i in range(1, len(data_set)):  # 合并不同类别的矩阵成一个，方便求均值
        data_all = np.vstack((data_all, data_set[i]))
    u = mean_of_matrix(data_all)
    print(u.shape)
    # print("u of all:")
    # print(u)
    ui = [mean_of_matrix(data) for data in data_set]
    # print("ui:")
    # print(ui)
    Sb = 0
    for i in ui:
        Sb = Sb + (i - u).T * (i - u)
    return Sb


def LDA(dataSet):
    # 计算所有数据的类内离散度
    Sw = 0
    for data in dataSet:
        Sw = Sw + compute_Sw(data)
    print("Sw:")
    print(Sw)

    # 再计算所有数据的类间离散度
    Sb = compute_Sb(dataSet)
    print("Sb:")
    print(Sb)

    # print("np.linalg.det(Sw):")
    # print(np.linalg.det(Sw))
    # print("np.linalg.det(Sb):")
    # print(np.linalg.det(Sb))

    # 在优化的LDA算法里，先求的是Sw和Sb的特征值和特征向量
    Sb_eig_value, Sb_vec = np.linalg.eig(Sb)
    print("Sb eig value:")
    print(Sb_eig_value)
    print("Sb vec:")
    print(Sb_vec)
    Sb_eig_index = np.argsort(-Sb_eig_value)
    Sw_eig_value, Sw_vec = np.linalg.eig(Sw)
    print("Sw eig value:")
    print(Sw_eig_value)
    print("Sw vec:")
    print(Sw_vec)
    Sw_eig_index = np.argsort(-Sw_eig_value)

    # 试验区
    index = np.argsort(-Sb_eig_value)
    Sb2 = 0
    for i in range(np.linalg.matrix_rank(Sb)):
        Sb2 = Sb2 + Sb_eig_value[index[i]] * Sb_vec[:, index[i]] * Sb_vec[:, index[i]].T
    print('Sb2')
    print(Sb2)

    # 新方法
    sum1 = 0
    for j in range(np.linalg.matrix_rank(Sb)):
        sum2 = 0
        for i in range(np.linalg.matrix_rank(Sw)):  # i的范围有待商榷
            sum2 = sum2 + Sb_eig_value[Sb_eig_index[j]] / Sw_eig_value[Sw_eig_index[i]] * (
                    Sw_vec[:, Sw_eig_index[i]].T * Sb_vec[:, Sb_eig_index[j]]).item() * Sw_vec[:, Sw_eig_index[i]]
        sum1 = sum1 + sum2 * Sb_vec[:, Sb_eig_index[j]].T

    eig_value, vec = np.linalg.eig(sum1)
    print("eig value:")
    print(eig_value)
    print("vec:")
    print(vec)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    print("index:")
    print(index)
    W = vec[:, index[0]]
    print("W:")
    print(W)

    # 旧方法求特征值和特征向量
    eig_value, vec = np.linalg.eig(np.mat(Sw).I * Sb)
    print("eig value:")
    print(eig_value)
    print("vec:")
    print(vec)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    print("index:")
    print(index)
    W = vec[:, index[0]]
    print("W:")
    print(W)
    return W


if __name__ == '__main__':
    dataSet = creatData(8, 2)
    # print(dataSet)

    # print(np.vstack((s1, s2)))
    # print(np.hstack((s1.T, s2.T)))
    # print(np.hstack((s1.T, s2.T)).tolist())
    # drawFigure(np.hstack((s1.T, s2.T)))
    LDA(dataSet)
    # drawFigure(dataSet, LDA(dataSet))  # 调用算法并画图
