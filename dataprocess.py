import random

from scipy.io import loadmat
import numpy as np
from sklearn import datasets


def get_data(data_name='orl', keepDim=False):
    """
    请自行修改数据集路径
    :param data_name:
    :param keepDim:
    :return: data, n_class, n_per_class, label
    """
    if data_name.lower() == 'orl':
        mat = loadmat('E:/机器学习/数据集/ORL4646.mat')  # (46， 46, 40*10)
        if keepDim is True:
            return mat["ORL4646"].reshape((2116, 400)).T.reshape((400, 46, 46)), 40, 10, None
        return mat["ORL4646"].reshape((2116, 400)).T, 40, 10, None
    elif data_name.lower() == 'yale':
        mat = loadmat('E:/机器学习/数据集/Yale5040165.mat')  # (50, 40, 15*11)
        if keepDim is True:
            return mat["Yale5040165"].reshape((2000, 165)).T.reshape((165, 50, 40)), 15, 11, None
        return mat["Yale5040165"].reshape((2000, 165)).T, 15, 11, None
    elif data_name.lower() == 'feret':
        mat = loadmat('E:/机器学习/数据集/FERET74040.mat')  # (40， 40, 200*7)
        return mat["FERET74040"].reshape((1600, 1400)).T, 200, 7, None
    elif data_name.lower() == 'ar':
        mat = loadmat('E:/机器学习/数据集/AR120p20s50by40.mat')  # (50, 40, 120*20)
        if keepDim is True:
            return mat["AR120p20s50by40"].reshape((2000, 2400)).T.reshape((2400, 50, 40)), 120, 20, None
        return mat["AR120p20s50by40"].reshape((2000, 2400)).T, 120, 20, None

    elif data_name.lower() == 'iris':
        # 从sklearn加载数据集
        iris = datasets.load_iris()
        return iris['data'], 3, 50, None
    elif data_name.lower() == 'coil':
        mat = loadmat('E:/机器学习/数据集/COIL100.mat')  # (50, 40, 120*20)
        return mat['COIL100'].T, 100, 72, mat['gnd'].reshape(-1, 1)
    elif data_name.lower() == 'minist':
        mat = loadmat('E:/机器学习/数据集/minist_image.mat')  # (28, 28, 60000)
        label = loadmat('E:/机器学习/数据集/minist_labels.mat')
        return mat['train_images'].reshape((28 * 28, 60000)).T, 10, None, label['train_labels'].reshape(-1, 1)


def getPointSet(n_class=2, n_one_class=10, dimension=2):
    data = []
    for i in range(0, n_class):
        temp = np.random.random([n_one_class, dimension]) * 10 + (10 + 10 * random.random()) * i
        j = random.choice(range(dimension))
        if random.random() < 0.5:
            temp[:, j] -= (10 + 10 * random.random())
        else:
            temp[:, j] += (10 + 10 * random.random())
        data.append(temp)
    # data = [np.mat(np.random.random([n_one_class, 2]) * 5 + random.uniform(0, 10) + 13 * i) for i in range(0,
    # n_class)]
    return np.array(data).reshape(n_one_class * n_class, -1)


def separateData(dataset, k, start, n_class, k_is_rate=False):  # k:每人训练的张数 start:训练集开始的地方
    """
    本分离器不适用于类别信息被打乱的训练集（当然强行用也行，就是容易错）
    :param dataset:
    :param k: 如果是占比，区间为(0, 1)
    :param start:
    :param n_class:
    :param k_is_rate:指定k是占比还是具体数量
    :return:
    """

    n = dataset.shape[0]  # 图片张数, ORL为400
    n1 = int(n // n_class)  # 每个人的张数
    if k_is_rate:
        if not 0 < k <= 1:
            raise Exception
        k = n1 * k
    shape1 = list(dataset.shape)
    shape1[0] = n_class * k
    shape2 = list(dataset.shape)
    shape2[0] = n_class * (n1 - k)

    # 格式!
    train_data = np.zeros(shape=shape1)
    train_data_index = np.zeros((n_class * k, 1))
    test_data = np.zeros(shape=shape2)
    test_data_index = np.zeros((n_class * (n1 - k), 1))
    train_num = 0  # 训练集的个数
    test_num = 0  # 测试集的个数
    for i in range(n_class):
        for j in range(n1):
            if j < k:
                train_data[train_num] = dataset[n1 * i + (start + j + n1) % n1]
                train_data_index[train_num] = i
                train_num += 1
            else:
                test_data[test_num] = dataset[n1 * i + (start + j + n1) % n1]
                test_data_index[test_num] = i
                test_num += 1

    return train_data, train_data_index, test_data, test_data_index
    # 分离测试集和训练集并给它们标号


# 按列计算均值, 返回均值向量
def mean_of_matrix(X):
    a = X.shape[0]
    tempX = 0
    for i in range(a):
        tempX += X[i]
    if len(tempX.shape) == 1:
        tempX = tempX.reshape((1, -1))
    return tempX / a


def get_normalize(X):
    u = mean_of_matrix(X)
    std = X.std(axis=0, ddof=1)
    for i in range(std.shape[0]):
        if std[i] == 0:
            std[i] = 1
    return (X - u) / std


def normalize_image_array(image_array):
    # N, D = image_array.shape

    numerator = image_array - np.expand_dims(np.mean(image_array, 1), 1)
    denominator = np.expand_dims(np.std(image_array, 1), 1)

    return numerator / (denominator + 1e-7)


if __name__ == '__main__':
    mat = loadmat('E:/机器学习/数据集/minist_labels.mat')  # (2, 40, 120*20)
    print(mat['train_labels'].shape)
