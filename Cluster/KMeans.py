from time import time

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from PCA.pca import pcaProject
from dataprocess import get_data, separateData, getPointSet

colors = ['lime', 'r', 'b', 'm', 'slategrey', 'c', 'y', 'k', 'sandybrown', 'g']
markers = ['o', 'x', '+', '^', '<', 'v', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
           'P', 'X']  # 圆圈， x, +, 正三角， 倒三角...
matplotlib.rcParams['axes.unicode_minus'] = False  # 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签


def distEuclidean(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)


def randCent(dataSet, k):  # 第一个中心点初始化
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros([k, n]))  # 创建 k 行 n列的全为0 的矩阵
    for j in range(n):
        minJ = np.min(dataSet[:, j])  # 获得第j 列的最小值
        rangeJ = float(np.max(dataSet[:, j]) - minJ)  # 得到最大值与最小值之间的范围
        # 获得输出为 K 行 1 列的数据，并且使其在数据集范围内
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def getCent(dataSet, k):
    """
    给KMeans++
    :param dataSet:
    :param k:
    :return:
    """
    print("no defined yet")


# 参数： dataSet 样本点， K 簇的个数
# disMeans 距离， 默认使用欧式距离， createCent 初始中心点的选取
def KMeans(dataSet, k, distFun=distEuclidean, createCent=randCent):
    m = np.shape(dataSet)[0]  # 得到行数，即为样本数
    clusterAssignment = np.mat(np.zeros([m, 2]))  # 创建 m 行 2 列的矩阵
    centroids = createCent(dataSet, k)  # 初始化 k 个中心点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf  # 初始设置值为无穷大
            minIndex = -1
            for j in range(k):
                #  j循环，先计算 k个中心点到1 个样本的距离，在进行i循环，计算得到k个中心点到全部样本点的距离
                if np.nan in centroids[j, :]:
                    distJ = np.inf
                else:
                    distJ = distFun(centroids[j, :], dataSet[i, :])
                if distJ < minDist:
                    minDist = distJ  # 更新 最小的距离
                    minIndex = j
            if clusterAssignment[i, 0] != minIndex:  # 如果中心点不变化的时候， 则终止循环
                clusterChanged = True
            clusterAssignment[i, :] = minIndex, minDist  # 将 index，k值中心点 和  最小距离存入到数组中
        # print(centroids)

        # 更换中心点的位置
        for cent in range(k):
            ptsInCluster = dataSet[np.nonzero(clusterAssignment[:, 0].A == cent)[0]]  # 分别找到属于k类的数据
            if ptsInCluster.size == 0:
                centroids[cent, :] = np.nan
            else:
                centroids[cent, :] = np.mean(ptsInCluster, axis=0)  # 得到更新后的中心点
    return centroids, clusterAssignment


def clusterTest(fun_name=KMeans, n_cluster=2, data_name='orl', k=5, kernel_type=None):
    """

    :param n_cluster: 聚类数目
    :param fun_name: K
    :param data_name:
    :param k: training sample percent
    :param kernel_type: none
    :return:
    """
    print('In cluster:\n' + 'cluster type: ' + fun_name.__name__)
    print('Kernel type: ' + str(kernel_type))
    print('In ' + data_name)
    print('K = %d' % k)

    # 加载数据集
    oldData, n_class, n_one_class, oldLabel = get_data(data_name)

    # 注意PCA处理
    # data = pcaProject(oldData, dimension=160)

    # ---------------取前面3个类-------------
    # n_class = 3
    # n = None
    # # PCA 处理
    # data = pcaProject(oldData[0:n_class * n_one_class], dimension=160)
    # # 不处理
    # # data = oldData[:n_class * n_one_class]
    # if oldLabel is not None:
    #     label = oldLabel[0:n_class * n_one_class]
    # ---------------------------------------

    # ---------------或者取前面n张-------------
    n = 100
    """
    修改张数
    """
    # PCA 处理0.791   0.797  0.626   0.99
    #         0.221   1.27   5.76
    data = pcaProject(oldData[0:n], dimension=120)
    # 不处理
    # data = oldData[:n]
    if oldLabel is not None:
        label = oldLabel[:n]
    else:
        label = None
    # ---------------------------------------

    # 生成点集
    # n_class = 3
    # n_one_class = 10
    # data = getPointSet(n_class=n_class, n_one_class=n_one_class, dimension=2)

    print(data.shape)
    accuracy = 0
    if n is None and n_one_class is not None:
        n_train = int(n_one_class * k / 10)
    elif n is not None:
        n_train = n
    else:
        print('something wrong...')
        return

    # 忘了这是干嘛的
    if n_train == 1:
        n_train = 2

    print('n of train set: %d' % n_train)
    # 聚类没有单独的投影矩阵给测试集进行准确度的预测，因此测试集可以去除
    train_data, train_data_index, test_data, test_data_index = separateData(data, 1, 0, n_class, k_is_rate=True)
    if oldLabel is not None:
        train_data_index, _, test_data_index, _ = separateData(label, 1, 0, n_class, k_is_rate=True)

    print(train_data.shape)

    t1 = time()
    centroids, clusterAssignment = fun_name(train_data, n_cluster, distFun=distEuclidean, createCent=randCent)
    t2 = time()
    print("运行时间：", t2 - t1)
    # 类别如果按顺序放
    # res = 0
    # for i in range(n_class):  # n_class：种类数 n_train：一类样本的个数
    #     a = clusterAssignment[i * n_train:(i + 1) * n_train, 0]  # label
    #     temp = max([np.sum(a == j) for j in set(a.flat)])  # if res == label
    #     print('temp' + str(temp))
    #     res += temp / n_train
    # res /= n_class

    # 不按顺序放
    res = 0
    for i in range(n_class):  # n_class：种类数 n_train：一类样本的个数
        a = np.array(
            [clusterAssignment[j, 0] for j in range(clusterAssignment.shape[0]) if train_data_index[j] == i])  # label
        temp = max([np.sum(a == j) for j in set(a.flat)])  # if res == label
        print('class ' + str(i) + ' num ' + str(temp) + ' of ' + str(a.size))
        res += temp / a.size
    res /= n_class
    print(res)

    # plotData = train_data
    plotData = pcaProject(oldData[:train_data.shape[0]], dimension=2)
    print('plotData.shape: ', plotData.shape)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    for i in range(plotData.shape[0]):
        t = int(clusterAssignment[i, 0])
        ax.scatter(plotData[i, 0], plotData[i, 1], s=30, c=colors[t], marker=markers[int(train_data_index[i])])
    plt.savefig('E:\\最优化方法\\实验\\test'+str(n)+'.png', dpi=1000)
    plt.show()


if __name__ == '__main__':
    clusterTest(fun_name=KMeans, n_cluster=10, data_name='minist', k=10, kernel_type=None)
    # b = np.array([[0, 4, 4], [2, 0, 3], [1, 3, 4]])
    # l = sorted([(np.sum(b == i), i) for i in set(b.flat)])
    # print(l)
    # mat = loadmat('E:/机器学习/数据集/COIL100.mat')
    # a = mat['COIL100'][:, 0]
    # print(a)
