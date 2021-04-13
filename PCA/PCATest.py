import time

import matplotlib
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from sklearn import datasets

from LDA import lda
from LDA.LDAtest1 import LDA_test, get_normalize
from PCA.B2DPCA import BTwoDPCA, twoDPCA
from PCA.pca import pca
from PCA.pca import mean_of_matrix
import numpy as np
import matplotlib.pyplot as plt

from dataprocess import get_data, separateData

colors = ['r', 'g', 'b', 'm', 'k', 'c', 'y', 'slategrey', 'sandybrown', 'lime']
matplotlib.rcParams['axes.unicode_minus'] = False  # 设置字体


def pcaKNN(train_mat, test_mat, train_data_index):
    num = 0
    result = np.zeros(test_mat.shape[0])
    # 方法一 KNN
    for i in range(test_mat.shape[0]):
        dist = [np.linalg.norm(train_mat[j] - test_mat[i]) for j in range(train_mat.shape[0])]

        # dist = np.abs(dist)
        # dist = np.sum(dist, axis=1)  # 也可用范数
        # dist = [np.linalg.norm(test_vec @ res_W[int(i // k)].T - train_mat[i].reshape(1, res_W[0].shape[0])) for i in
        #         range(train_mat.shape[0])]
        index = np.argsort(dist)  # 将测试图片与训练图距离从小到大排序，得到在训练图中的位置标号
        # tag = np.zeros(int(np.max(train_data_index) + 1))  # 创建所有人的票数计数器，初始为零
        # for i in range(3):
        #     tag[int(train_data_index[index[i]])] += 1  # 根据标号找到类别并投票
        # if np.max(tag) == 1:
        #     result[num] = int(train_data_index[index[0]])  # 如果得到的不同类别票数一样,找到距离最近的
        # else:
        #     result[num] = np.argmax(tag)  # 找到票数最多的，作为结果
        result[num] = int(train_data_index[index[0]])  # 找到距离最近的，作为结果
        num += 1
        # 方法二
        # dist = [np.linalg.norm((test_vec - train_mat[i])@ res_W[i].T) for i in range(train_mat.shape[0])]
    return result.reshape((-1, 1))


def get_mat(res_W, train_data, test_data, d):
    # tempW[i](d,train_data.shape[1])
    # 对所有样本投影
    # 第一个方法

    train_mat = np.zeros((train_data.shape[0], d))
    for i in range(train_data.shape[0]):
        train_mat[i] = train_data[i].reshape((1, train_data.shape[1])) @ res_W.T
    test_mat = np.zeros((test_data.shape[0], d))
    for i in range(test_data.shape[0]):
        test_mat[i] = test_data[i].reshape((1, test_data.shape[1])) @ res_W.T
    # 第二个方法，获得平均脸的投影
    # train_mat = np.zeros((n_of_man, train_data.shape[1]))
    # for i in range(n_of_man):
    #     u = mean_of_matrix(np.array(train_data[i * k:(i + 1) * k]))
    #     u = u.reshape((1, u.shape[0]))
    #     train_mat[i] = u  # @ res_W[i].T
    return train_mat, test_mat


def pca_test(k=8, data_name='orl', n_used_class=None, dimension=120):
    print('K = ' + str(k) + ':')
    data, n_class, n_one_class, _ = get_data(data_name)
    data = data / np.max(data)

    if n_used_class is not None:
        # 选取一部分
        n_class = n_used_class
        data = data[:n_class * n_one_class]

    accuracy = 0
    total_time = 0

    n_train = int(n_one_class * k / 10)
    if n_train == 1:
        n_train = 2
    print('n of train set: %d' % n_train)

    for i in range(n_one_class):
        train_data, train_data_index, test_data, test_data_index = separateData(data, n_train, i, n_class)
        # train_data_norm = get_normalize(train_data)
        # test_data_norm = get_normalize(test_data)
        # train_data_norm = train_data
        # test_data_norm = test_data
        time1 = time.time()
        pca_W = pca(train_data)  # 特征矩阵
        # lda_W = lda.LDATrain(train_data, n_class, k)  # 特征矩阵
        # lda_W = lda.LDATrain(train_data, n_class, k)  # 特征矩阵

        # matplot show image
        # img = train_data[0].reshape((46, 46))
        # fig = plt.figure()
        # for j in range(4):
        #     # PCA
        #     img = pca_W[j].reshape((46, 46))
        #     plt.subplot(2, 4, j + 1)
        #     plt.axis('off')
        #     plt.xticks([])  # 去掉横坐标值
        #     plt.yticks([])  # 去掉纵坐标值
        #     plt.title('PCA')
        #     plt.imshow(img, cmap='Greys_r')
        #     # LDA
        #     img = lda_W[j].reshape((46, 46))
        #     plt.subplot(2, 4, j + 5)
        #     plt.axis('off')
        #     plt.xticks([])  # 去掉横坐标值
        #     plt.yticks([])  # 去掉纵坐标值
        #     plt.title('LDA')
        #     plt.imshow(img, cmap='Greys_r')
        # plt.tight_layout()
        # plt.savefig('E:\\机器学习\\实验报告\\实验2\\eigenface与fisherface(pinv).png')
        # plt.show()

        # showImage
        # 我不用这一部分，仅作参考和日后学习
        # image1 = Image.new("RGB", (46*8, 132), 'white')
        # size = 46
        # x = 0
        # y = 0
        # img = Image.fromarray(train_data[0].reshape((size, size)))
        # font = ImageFont.truetype("simhei.ttf", 5, encoding="utf-8")
        # draw = ImageDraw.Draw(image1)  # 可以理解为画笔工具
        # draw.text((x, y), 'original', "black")
        # for j in range(8):
        #     image1.paste(img, (x+j*46, y+20))
        # image1.show()

        # continue matplot image show
        # for j in range(8):
        #     plt.subplot(2, 8, j + 9)
        #     plt.axis('off')
        #     plt.xticks([])  # 去掉横坐标值
        #     plt.yticks([])  # 去掉纵坐标值
        #     plt.title('dimension = '+str((j+1)*20))
        #     img = (train_data[0].reshape((1, train_data.shape[1]))-mu) @ np.array(W[:(j+1)*20]).T @ np.array(W[:(j+1)*20]) + mu
        #     img = (img.reshape((46, 46))).real
        #     plt.imshow(img, cmap='Greys_r')
        # plt.tight_layout()
        # plt.savefig('E:\\机器学习\\实验报告\\face_in_dif_D.png')
        # plt.show()
        # break

        #     测试识别率
        res_W = np.array(pca_W[:dimension])  # get dimension dimension
        train_mat, test_mat = get_mat(res_W, train_data, test_data, dimension)
        res = np.mean(test_data_index == pcaKNN(train_mat, test_mat, train_data_index))
        print("res %d " % (i + 1))
        print(res)
        accuracy += res
        time2 = time.time()
        total_time = time2 - time1
    total_time /= n_one_class
    print('total_time:')
    print(total_time)
    accuracy /= n_one_class
    print('accuracy:')
    print(accuracy)
    return accuracy


def pca_test_d(d, k=7):
    # mat = loadmat('E:/机器学习/数据集/Yale5040165.mat')  # (50, 40, 165)
    mat = loadmat('E:/机器学习/数据集/ORL4646.mat')  # (46， 46, 400)
    data = mat["ORL4646"].reshape((2116, 400))
    # data = mat["Yale5040165"].reshape((2000, 165))
    n_of_man = 40
    accuracy = 0
    res_of_d = np.zeros(len(d))
    for i in range(10):
        train_data, train_data_index, test_data, test_data_index = separateData(data.T, k, i, n_of_man)
        mu = mean_of_matrix(train_data)
        W = pca(train_data)  # 特征矩阵
        num = 0
        for j in d:
            res_W = np.array(W[:j])  # get j dimension,which is j, :
            train_mat, test_mat = get_mat(res_W, train_data, test_data, j)
            res = np.mean(test_data_index == pcaKNN(train_mat, test_mat, train_data_index))
            print("res %d , %d d" % ((i + 1), j))
            print(res)
            res_of_d[num] += res
            num += 1
    res_of_d /= 10
    print('result:----------------------')
    print(res_of_d)
    return res_of_d


def PCA_draw_fig1():
    print('--------------------------------------------------')
    print('For different k:')
    print('--------------------------------------------------')
    plt.figure(1)
    x = [i for i in range(2, 10)]  # k的取值
    y1 = [pca_test(x[i]) for i in range(len(x))]
    y2 = [LDA_test(x[i]) for i in range(len(x))]
    print("k of PCA max accuracy: %d" % x[np.argmax(y1)])
    print("k of LDA max accuracy: %d" % x[np.argmax(y2)])
    plt.plot(x, y1, label='PCA', c='r')
    plt.plot(x, y2, label='LDA', c='y')
    plt.title(u"ORL人脸库 Different k")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('E:\\机器学习\\实验报告\\实验2\\ORLDifferent_K.png')
    plt.show()
    print(x)
    print(y1)
    print(y2)


def PCA_draw_fig2():
    print('--------------------------------------------------')
    print('For different d:')
    print('--------------------------------------------------')

    name_list = ['B2DPCA', 'PCA']
    # name_list = ['regularized', 'b2dpca', 'pca']
    num = 10  # d最多的数
    plt.figure(2)
    x = [i * i for i in range(1, num + 1)]  # d的个数
    y = []
    # 改数据集记得改代码1,2,3
    file_name = 'yale_Different_d_2'
    data_name = 'yale'
    f = open("E:\\专业研究英语\\fig\\" + file_name + ".txt", "w+")  # 代码1
    f.writelines('d from 1 to ' + str(100))
    for i in range(len(name_list)):
        if name_list[i].lower() == 'pca':
            y.append([pca_test(k=6, data_name=data_name, dimension=x[j]) for j in range(len(x))])
        elif name_list[i].lower() == 'b2dpca':
            y.append([B2DPCA_test(k=6, data_name=data_name, dimension=int(np.sqrt(x[j])),
                                  fun_name='b2dpca') for j in range(len(x))])
        elif name_list[i].lower() == '2dpca':
            y.append([B2DPCA_test(k=6, data_name=data_name, dimension=x[j], fun_name='2dpca') for j in range(len(x))])
        else:
            y.append([LDA_test(6, fun_type=name_list[i], data_name=data_name, dimension=x[j]) for j in range(len(x))])
        print("d of max accuracy in " + name_list[i] + ": %d" % x[np.argmax(y[i])])
        plt.plot(x, y[i], label=name_list[i], c=colors[i])
        f.write(name_list[i] + ':\n' + str(y[i]) + '\n')
    plt.title(data_name.upper())  # 代码2
    plt.xlabel("Dimensions")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('E:\\专业研究英语\\fig\\' + file_name + '.png', dpi=600)  # 代码3
    plt.show()
    f.close()


def draw_scatter():
    n_type = 3
    mat = loadmat('E:/机器学习/数据集/ORL4646.mat')  # (46， 46, 40*10)
    data = mat["ORL4646"].reshape((2116, 400))
    # mat = loadmat('E:/机器学习/数据集/Yale5040165.mat')  # (50, 40, 15*11)
    # data = mat["Yale5040165"].reshape((2000, 165))
    # mat = loadmat('E:/机器学习/数据集/AR120p20s50by40.mat')  # (50, 40, 120*20)
    # data = mat["AR120p20s50by40"].reshape((2000, 2400))
    data = data.T
    n_of_man = 40
    d = 2
    k = 10
    # 根据设置获得不同类型的矩阵
    pca_W = pca(data)
    # lda_W = lda.LDATrain(data, n_of_man, k, type='msd')  # 特征矩阵

    # 2d
    res_W = np.array(pca_W[:2])  # get d dimension d*n
    res_mat = data @ res_W.T  # m*d
    res_mat, res_mat = get_mat(res_W, data, data, d)
    fig = plt.figure(1)
    plt.xlim(-7000, -1000)
    plt.ylim(-6000, 1000)
    # ax = fig.add_subplot(211)
    # ax = fig.gca(projection='3d')
    # x = res_mat[:-1, 0]
    # y = res_mat[:-1, 1]
    # plt.scatter(x, y, c=colors[0])
    # plt.show()
    # return
    for i in range(n_type):
        x = res_mat[k * i:k * (i + 1), 0]
        y = res_mat[k * i:k * (i + 1), 1]
        plt.scatter(x, y, c=colors[i])
    plt.savefig('E:\\机器学习\\实验报告\\实验2\\ORL_PCA投影2d.png')
    # plt.savefig('E:\\机器学习\\实验报告\\实验2\\ORL_LDA投影2d.png')
    # plt.savefig('E:\\机器学习\\实验报告\\实验2\\AR_LDA投影2d.png')
    plt.show()
    # 3d
    res_W = np.array(pca_W[:3])  # get d dimension d*n
    res_mat = data @ res_W.T  # m*d
    res_mat = res_mat.real
    fig = plt.figure(2)
    plt.xlim(-7000, -1000)
    plt.ylim(-6000, 1000)

    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    # ax.set_zlim3d(-1000, 1000)
    for i in range(n_type):
        x = res_mat[k * i:k * (i + 1), 0]
        y = res_mat[k * i:k * (i + 1), 1]
        z = res_mat[k * i:k * (i + 1), 2]
        ax.scatter(x, y, z, c=colors[i])
    plt.savefig('E:\\机器学习\\实验报告\\实验2\\ORL_PCA投影3d.png')
    # plt.savefig('E:\\机器学习\\实验报告\\实验2\\ORL_LDA投影3d.png')
    # plt.savefig('E:\\机器学习\\实验报告\\实验2\\AR_LDA投影3d.png')
    plt.show()
    # ax.set_xlabel
    # ax.set_ylabel
    # ax.set_xlabel


def B2DPCA_test(k=8, data_name='orl', dimension=5, fun_name='B2DPCA'):
    """

    :param k:
    :param data_name:
    :param dimension:
    :param fun_name: B2DPCA, 2DPCA
    :return:
    """
    print('K = ' + str(k) + ':')
    data, n_class, n_one_class, _ = get_data(data_name, keepDim=True)
    # data = data / np.max(data)
    accuracy = 0
    total_time = 0

    n_train = int(n_one_class * k / 10)
    if n_train == 1:
        n_train = 2
    print('n of train set: %d' % n_train)

    for i in range(n_one_class):
        train_data, train_data_index, test_data, test_data_index = separateData(data, n_train, i, n_class)

        time1 = time.time()
        if fun_name.lower() == 'b2dpca':
            pca_W1, pca_W2 = BTwoDPCA(train_data, dimension)  # 特征矩阵
        elif fun_name.lower() == '2dpca':
            pca_W1 = twoDPCA(train_data, dimension)
        train_mat = []
        test_mat = []

        for j in range(train_data.shape[0]):
            if fun_name.lower() == 'b2dpca':
                temp1 = (train_data[j, :, :] @ pca_W1).T @ pca_W2  # (b, c) @ (c, k) = (b, k)
            else:
                temp1 = train_data[j, :, :] @ pca_W1  # (b, c) @ (c, k) = (b, k)

            train_mat.append(temp1.T)  # (k, b)  1

        for j in range(test_data.shape[0]):
            if fun_name.lower() == 'b2dpca':
                temp1 = (test_data[j, :, :] @ pca_W1).T @ pca_W2  # (b, c) @ (c, k) = (b, k)
            else:
                temp1 = test_data[j, :, :] @ pca_W1  # (b, c) @ (c, k) = (b, k)

            test_mat.append(temp1.T)  # (k, b)  1

        train_mat = np.array(train_mat)
        test_mat = np.array(test_mat)

        res = np.mean(test_data_index == pcaKNN(train_mat, test_mat, train_data_index))
        print("res %d " % (i + 1))
        print(res)
        accuracy += res
        time2 = time.time()
        total_time = time2 - time1
    total_time /= n_one_class
    print('total_time:')
    print(total_time)
    accuracy /= n_one_class
    print('accuracy:')
    print(accuracy)
    return accuracy


if __name__ == '__main__':
    # draw_scatter()
    # PCA_draw_fig1()
    # PCA_draw_fig2()
    # pca_test(k=6, data_name='ar')
    max_acc = 0
    max_d = 0
    for d in range(1, 30):
        acc = B2DPCA_test(k=6, data_name='ar', dimension=d, fun_name='b2dpca')
        if acc > max_acc:
            max_acc = acc
            max_d = d
    print("The best is " + str(max_d) + "with acc" + str(max_acc))
    # a= np.ones(shape=(2,2))*2
    # b= np.ones(shape=(2,2))
    # c=np.ones(shape=(2,2))
    # print(a@(b-c))
