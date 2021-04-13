import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LDA.creatData import creatData
from LDA.draw import drawFigure
from dataprocess import mean_of_matrix

lam = 0.1
alpha = 0.5


# 求一个人的所有照片离散度
def compute_Sw(data):
    Sw = 0
    u = mean_of_matrix(data)
    for i in range(0, data.shape[0]):
        Sw = Sw + (data[i] - u).T @ (data[i] - u)
    return Sw


# 改进后的计算类间离散度
def compute_Sb(data_set):
    all_data = data_set[0]
    for i in range(1, data_set.shape[0]):  # 合并不同类别的矩阵成一个，方便求均值
        all_data = np.vstack((all_data, data_set[i]))
    u = mean_of_matrix(all_data)
    ui = [mean_of_matrix(data) for data in data_set]  # 求出每个类的类中心
    Sb = 0
    for i in ui:
        Sb = Sb + (i - u).T @ (i - u)
    return Sb


# 分离出训练的样本
def LDATrain(all_data, n_class, n_one_class, i_of_e=0, fun_name=None):  # 数据集data_set，n个类别以及每个类别的k个数据, i_of_e 最终取e值的i值
    # 这里的data_set中每个data都要不同类
    # 计算所有数据的类内离散度
    data_set = np.zeros((n_class, n_one_class, all_data.shape[1]))
    # 将编号为i的人的照片放到一起
    num = 0
    for i in range(n_class):
        for j in range(n_one_class):
            data_set[i, j] = all_data[num]
            num += 1
    if fun_name is None or fun_name.lower() == 'None':
        return LDA(data_set, i_of_e)
    elif fun_name.lower() == 'classical':
        return classical_LDA(data_set)
    elif fun_name.lower() == 'regularized':
        return regularized_LDA(data_set)
    elif fun_name.lower() == 'st':
        return St_LDA(all_data, data_set)
    elif fun_name.lower() == 'msd':
        return MSD(data_set)
    return None


def MSD(data_set):
    print('In MSD:')
    Sw = 0
    for data in data_set:  # 求所有照片类内离散度
        Sw = Sw + compute_Sw(data)
    # 再计算所有数据的类间离散度
    Sb = compute_Sb(data_set)
    eig_value, vec = np.linalg.eigh(Sb - alpha * Sw)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    vec = vec.T
    W = [vec[index[i]] for i in range(len(index))]  # 行列变一下方便后面处理的统一
    return np.array(W)


def St_LDA(all_data, data_set):
    print('In St LDA:')
    u = mean_of_matrix(all_data)
    St = (all_data - u).T @ (all_data - u)
    Sb = compute_Sb(data_set)
    eig_value, vec = np.linalg.eigh(np.linalg.inv(St + lam * np.eye(St.shape[0])) @ Sb)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    vec = vec.T
    W = [vec[index[i]] for i in range(len(index))]  # 行列变一下方便后面处理的统一
    return np.array(W)


def classical_LDA(data_set):
    print('In classical LDA:')
    Sw = 0
    for data in data_set:  # 求所有照片类内离散度
        Sw = Sw + compute_Sw(data)
    # 再计算所有数据的类间离散度
    Sb = compute_Sb(data_set)
    # 传统方法就是直接求逆，参考下面print里的式子，但是要有逆
    # 这里我们求伪逆
    eig_value, vec = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)
    # print('np.mat(Sw).I * Sb:')
    # print(np.mat(Sw).I @ Sb)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    vec = vec.T
    W = [vec[index[i]] for i in range(len(index))]  # 行列变一下方便后面处理的统一
    return np.array(W)


def regularized_LDA(data_set):
    print('In regularized LDA:')
    Sw = 0
    for data in data_set:  # 求所有照片类内离散度
        Sw = Sw + compute_Sw(data)
    # 再计算所有数据的类间离散度
    Sb = compute_Sb(data_set)
    eig_value, vec = np.linalg.eigh(np.linalg.inv(Sw + lam * np.eye(Sw.shape[0])) @ Sb)
    # eig_value, vec = np.linalg.eigh(np.linalg.solve(Sw + lam * np.eye(Sw.shape[0]), Sb))
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    vec = vec.T
    W = [vec[index[i]] for i in range(len(index))]  # 行列变一下方便后面处理的统一
    return np.array(W)


def LDA(data_set, i_of_e):
    """
    这个是论文里的方法
    :param data_set: data_set中每个data都要不同人
    :param i_of_e: 决定下面的e值的i
    :return: 投影矩阵
    """
    Sw = 0
    for data in data_set:  # 求所有照片类内离散度
        Sw = Sw + compute_Sw(data)

    # 再计算所有数据的类间离散度
    Sb = compute_Sb(data_set)

    # 新方法
    # 在优化的LDA算法里，先求的是Sw和Sb的特征值和特征向量
    Sb_eig_value, Sb_vec = np.linalg.eigh(Sb)
    Sb_eig_index = np.argsort(-Sb_eig_value)
    Sw_eig_value, Sw_vec = np.linalg.eigh(Sw)
    Sw_eig_index = np.argsort(-Sw_eig_value)
    a = np.linalg.matrix_rank(Sw)
    b = np.linalg.matrix_rank(Sb + Sw)
    c1 = (b - a) / 4
    c2 = (b - 100) / 4
    s = a / 100
    t = b / 100
    e = np.linalg.matrix_rank(Sw)

    # 在这里修改i，获得最大的e 1<i<5
    # i = 2
    print("i = %d" % i_of_e)
    if i_of_e == 0 or i_of_e == -1:
        pass
    elif t <= 1:
        e = b - (i_of_e - 1) * 2
    elif t <= 1.5:
        if s <= 1:
            e = a + (i_of_e - 1) * c1
        else:
            e = 101 + (i_of_e - 1) * c2
    elif t <= 2:
        e = 102 + (i_of_e - 1) * 13
    else:
        e = 163 + (i_of_e - 1) * 13
    print("e = ", end="")
    print(e)
    sum1 = 0
    for j in range(np.linalg.matrix_rank(Sb)):
        sum2 = 0
        for i in range(int(e)):  # i的范围与e相关
            sum2 = sum2 + Sb_eig_value[Sb_eig_index[j]] / Sw_eig_value[Sw_eig_index[i]] * (
                    Sw_vec[:, Sw_eig_index[i]].T @ Sb_vec[:, Sb_eig_index[j]]).item() * \
                   Sw_vec[:, Sw_eig_index[i]].reshape((Sw_vec.shape[0], 1))
        sum1 = sum1 + sum2 @ Sb_vec[:, Sb_eig_index[j]].reshape(Sb_vec.shape[0], 1).T

    # 传统方法就是直接求逆，参考下面print里的式子，但是要有逆
    # print('np.mat(Sw).I * Sb:')
    # print(np.mat(Sw).I @ Sb)

    # 求特征值和特征向量
    print('rank sum1')
    print(np.linalg.matrix_rank(sum1))
    eig_value, vec = np.linalg.eigh(sum1)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    print('eig_value:')
    print(eig_value)
    print('num of vec: ', end="")
    print(len(index))
    vec = vec.T
    W = [vec[index[i]] for i in range(len(index))]
    return np.array(W)


def LDAProject(all_data, n_class, n_one_class, i_of_e=0, fun_name=None, dimension=None, LDA_mat=None):
    best_d = n_class - 1
    if LDA_mat is None:
        LDA_mat = LDATrain(all_data, n_class, n_one_class, i_of_e=i_of_e, fun_name=fun_name)
    if dimension is not None and 0 < dimension < n_class:
        best_d = dimension
    tempW = np.array(LDA_mat[:best_d, :])
    return all_data @ tempW.T


if __name__ == '__main__':
    dataSet = creatData(8, 2)
    print(dataSet)

    # print(np.vstack((s1, s2)))
    # print(np.hstack((s1.T, s2.T)))
    # print(np.hstack((s1.T, s2.T)).tolist())
    # drawFigure(np.hstack((s1.T, s2.T)))
    # LDA(dataSet)
    # drawFigure(dataSet, LDA(dataSet))  # 调用算法并画图
