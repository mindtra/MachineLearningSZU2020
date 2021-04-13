# -*- coding: utf-8 -*-
from scipy.io import loadmat
import LDA.lda as lda
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import wraps
import matplotlib

from dataprocess import get_data, separateData, get_normalize

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
colors = ['r', 'g', 'b', 'm', 'k', 'c', 'y', 'slategrey', 'sandybrown', 'lime']
matplotlib.rcParams['axes.unicode_minus'] = False  # 设置字体


def time_count(fn):
    @wraps(fn)
    def measure_time(*arg, **kwargs):
        t1 = time.time()
        fn(*arg, **kwargs)
        t2 = time.time()
        res = t2 - t1
        print(f"@timefn: {fn.__name__} took {res: .5f} s")
        return res

    return measure_time


def get_mat(w, train_data, test_data):
    # 对所有样本投影
    # 第一个方法
    train_mat = np.zeros((train_data.shape[0], w.shape[0]))
    for i in range(train_data.shape[0]):
        train_mat[i] = train_data[i].reshape((1, train_data.shape[1])) @ w.T
    test_mat = np.zeros((test_data.shape[0], w.shape[0]))
    for i in range(test_data.shape[0]):
        test_mat[i] = test_data[i].reshape((1, test_data.shape[1])) @ w.T
    # print(train_mat.shape)
    # print(test_mat.shape)
    return train_mat, test_mat


def KNN(train_mat, test_mat, train_data_index, k=3):
    num = 0
    result = np.zeros(test_mat.shape[0])
    for test_vec in test_mat:
        dist = [np.linalg.norm(test_vec - train_vec) for train_vec in train_mat]
        index = np.argsort(dist)  # 将测试图片与训练图距离从小到大排序，得到在训练图中的位置标号
        tag = np.zeros(int(np.max(train_data_index) + 1))  # 创建所有人的票数计数器，初始为零
        for i in range(3):
            tag[int(train_data_index[index[i]])] += 1  # 根据标号找到类别并投票
        if np.max(tag) == 1:
            result[num] = int(train_data_index[index[0]])  # 如果得到的不同类别票数一样,找到距离最近的
        else:
            result[num] = np.argmax(tag)  # 找到票数最多的，作为结果
        # result[num] = int(train_data_index[np.argmin(dist)])  # 找到距离最近的，作为结果
        num += 1
    return result.reshape((-1, 1))


def LDA_test(k=2, i_of_e=0, fun_type=None, data_name='orl', n_used_class=None, dimension=None):
    # k:训练集个数; i_of_e 最终取e值的i值, 0或-1为不取e值
    print('K = ' + str(k) + ':')
    data, n_class, n_one_class, _ = get_data(data_name)

    if n_used_class is not None:
        # 选取一部分
        n_class = n_used_class
        data = data[:n_class * n_one_class]

    accuracy = 0
    total_time = 0

    best_d = n_class - 1
    if dimension is not None and 0 < dimension < n_class:
        best_d = dimension
    n_train = int(n_one_class * k / 10)
    if n_train == 1:
        n_train = 2
    print('n of train set: %d' % n_train)

    for i in range(n_one_class):
        train_data, train_data_index, test_data, test_data_index = separateData(data, n_train, i, n_class)

        train_data_norm = get_normalize(train_data)
        test_data_norm = get_normalize(test_data)

        time1 = time.time()
        W = lda.LDATrain(train_data_norm, n_class, n_train, i_of_e, fun_type)
        tempW = np.array(W[:best_d, :])  # get 1 to max_d dimension
        train_mat, test_mat = get_mat(tempW, train_data_norm, test_data_norm)
        res = np.mean(test_data_index == KNN(train_mat, test_mat, train_data_index, n_train))
        print("res %d" % (i + 1))
        print(res)
        accuracy += res
        time2 = time.time()
        total_time += time2 - time1
    accuracy /= n_one_class
    total_time /= n_one_class
    print('total_time:')
    print(total_time)
    print('accuracy:')
    print(accuracy)
    return accuracy


def LDA_test_4_d(max_d):  # 最大投影d维
    k = 6
    i_of_e = 0
    mat = loadmat('E:/机器学习/数据集/Yale5040165.mat')  # (50, 40, 15*11)
    # mat = loadmat('E:/机器学习/数据集/ORL4646.mat')  # (46， 46, 40*10)
    data = mat["Yale5040165"].reshape((2000, 165))
    n_of_man = 15
    res_of_d = np.zeros(max_d)
    for i in range(10):
        print('in No. %d' % (i + 1))
        train_data, train_data_index, test_data, test_data_index = separateData(data.T, k, i, n_of_man)
        # print('shape of dataset:')
        # print(train_data.shape)
        # print(train_data_index.shape)
        # print(test_data.shape)
        # print(test_data_index.shape)

        W = lda.LDATrain(train_data, n_of_man, k, i_of_e)
        for j in range(max_d):
            tempW = np.array(W[:j + 1, :])  # get j dimension,which is j, :
            print(tempW.shape)
            train_mat, test_mat = get_mat(tempW, train_data, test_data)
            res = np.mean(test_data_index == KNN(train_mat, test_mat, train_data_index, k))
            print("res %d , %d d" % ((i + 1), j + 1))
            print(res)
            res_of_d[j] += res
    res_of_d /= 10
    return res_of_d


def LDA_draw_fig1():
    print('--------------------------------------------------')
    print('For different i:')
    print('--------------------------------------------------')
    plt.figure(1)
    x = [i for i in range(0, 15)]  # i的取值
    y = [LDA_test(6, x[i]) for i in range(len(x))]
    print("i of max accuracy: %d" % x[np.argmax(y)])
    plt.plot(x, y, label='Different i')
    plt.title(u"Yale人脸库 k=6")
    plt.xlabel("i")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('E:\\PythonProject\\Machine learning\\LDA\\YaleK=6Different_i.png')
    plt.show()


def LDA_draw_fig2():
    print('--------------------------------------------------')
    print('For different d:')
    print('--------------------------------------------------')
    plt.figure(2)
    x = [i for i in range(0, 200)]  # d的个数
    y = LDA_test_4_d(200)  # 测试 d
    best_d = x[np.argmax(y)]
    print("d of max accuracy: %d" % best_d)
    plt.plot(x, y, label='Different d')
    plt.title(u"Yale人脸库 k=6")
    plt.xlabel("d")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('E:\\PythonProject\\Machine learning\\LDA\\YaleDifferentD.png')
    plt.show()
    return best_d


def LDA_draw_fig3():
    from PCA.PCATest import pca_test, B2DPCA_test
    print('--------------------------------------------------')
    print('For different K:')
    print('--------------------------------------------------')
    name_list = ['regularized', 'b2dpca', 'pca']
    num = 8  # k最多的张数
    plt.figure(3)
    x = [i for i in range(4, num + 1)]  # k的个数
    y = []
    # 改数据集记得改代码1,2,3
    file_name = 'yale_Different_K'
    data_name = 'yale'
    f = open("E:\\专业研究英语\\fig\\" + file_name + ".txt", "w+")  # 代码1
    f.write('K from 4 to 8')
    for i in range(len(name_list)):
        if name_list[i] == 'pca':
            y.append([pca_test(x[j], data_name=data_name) for j in range(len(x))])
        elif name_list[i] == 'b2dpca':
            y.append([B2DPCA_test(k=x[j], data_name=data_name, dimension=20, fun_name='b2dpca') for j in range(len(x))])
        else:
            y.append([LDA_test(x[j], data_name=data_name, fun_type=name_list[i]) for j in range(len(x))])
        print("k of max accuracy in " + name_list[i] + ": %d" % x[np.argmax(y[i])])
        plt.plot(x, y[i], label=name_list[i], c=colors[i])
        f.write(name_list[i] + ':\n' + str(y[i]) + '\n')
    plt.title(data_name.upper())  # 代码2
    plt.xlabel("numbers of training data per class")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('E:\\专业研究英语\\fig\\' + file_name + '.png', dpi=600)  # 代码3
    plt.show()
    f.close()


# 记得画特征向量不同个时的图像
if __name__ == '__main__':
    # LDA_draw_fig1()
    LDA_draw_fig3()
    # LDA_test(k=5, i_of_e=0, fun_type='MSD', data_name='ar', n_used_class=10, dimension=None)
    # best_d = LDA_draw_fig2()
    # LDA_draw_fig3()
    # PCA，LDA，线性回归不同的K
    # 线性回归方法在不同K时比较不同方法的识别率
    # LDA_test(5, 3)
    # mat = loadmat('E:/机器学习/数据集/Yale5040165.mat')
    # print(mat)
    # print(mat['Yale5040165'].shape)
    # print(mat['gnd'].shape)
    # y=[1,2,3]
    # print('lalala'+str(y))
