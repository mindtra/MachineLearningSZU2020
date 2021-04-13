# coding=utf-8
import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn import datasets

from LDA.lda import LDAProject, LDATrain
from PCA.pca import pcaProject
from dataprocess import get_data, separateData
from kernel import kernel_fun

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
colors = ['r', 'g', 'b', 'm', 'k', 'c', 'y', 'slategrey', 'sandybrown', 'lime']
matplotlib.rcParams['axes.unicode_minus'] = False  # 设置字体


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_mat, class_label, C=0.6, toler=0.001, max_iter=10, kernel='linear', sigma=1.0):
    # 循环外的初始化工作
    data_mat = np.mat(data_mat)
    label_mat = np.mat(class_label)
    b = 0
    m, n = np.shape(data_mat)
    alphas = np.zeros((m, 1))
    n_iter = 0
    count = 0
    while n_iter < max_iter:
        # 内循环的初始化工作
        alpha_pairs_changed = 0
        count += 1
        for i in range(m):
            # 第一小段代码
            # 计算误差
            KXXi = kernel_fun(data_mat, data_mat[i, :], kernel=kernel, sigma=sigma)
            f_xi = np.multiply(alphas, label_mat).T @ KXXi + b
            Ei = f_xi - float(label_mat[i])

            # KKT条件，若yi*(w^T * x +b)-1<0 则 ai=C  若yi*(w^T * x +b)-1>0 则 ai=0
            if ((label_mat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((label_mat[i] * Ei > toler) and (alphas[i] > 0)):
                # 筛选出需要修改的α
                """
                代码中的E_i为得到的f_xi与实际的WTX+b之间的误差，而这个误差大有说道，
                最完美的情况是若数据点是支持向量，则将数据点带入WTX+b应该等于-1或+1，
                即该数据点应该在虚线WTX+b+1=0或虚线WTX+b−1=0上，而如果该
                数据点的类标签yi和误差E_i的乘积大于容错率，说明该数据点属于+1类
                且在WTX+b−1=0的上侧，或者该数据点属于-1类且在WTX+b+1=0的下侧。
                也就是说该点在属于自己的那一类的群体里，但是太靠后了且靠后的超过限度了(即大于容错率)，
                根本无法作为支持向量，而这个点对应的α值又大于0，说明这明明就不该是靠后站的，
                那么就要调整这个数据点的α了。而另一种情况是该数据点的类标签yi和误差E_i的
                乘积小于负容错率，说明该点太靠前了已经处于虚线WTX+b+1和虚线WTX+b−1之间了，
                且该点的α值又小于常数C，说明该点也不应该是靠前的，那就要调整这个数据点的α了
                """
                j = select_jrand(i, m)  # 随机选出另一个α作为辅助筛选
                # WT_j = np.dot(np.multiply(alphas, label_mat).T, data_mat)
                # f_xj = float(np.dot(WT_j, data_mat[j, :].T)) + b
                KXXj = kernel_fun(data_mat, data_mat[j, :], kernel=kernel, sigma=sigma)
                f_xj = float(np.multiply(alphas, label_mat).T @ KXXj) + b
                Ej = f_xj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # 第二小段代码
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])  # 寻找α的可行域
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if H == L:
                    """
                    如果这个j让L=H,i和j这两个样本是同一类别，且ai=aj=0或ai=aj=C，或者不同类别，aj=C且ai=0
                    当同类别时 ai+aj = 常数 ai是不满足KKT的，假设ai=0,需增大它，那么就得减少aj，aj已经是0了，不能最小了，所以此情况不允许发生
                    当不同类别时 ai-aj=常数，ai是不满足KKT的，ai=0,aj=C,ai需增大，它则aj也会变大，但是aj已经是C的不能再大了，故此情况不允许
                    """
                    continue

                # 第三小段代码
                Kij = kernel_fun(data_mat[i, :], data_mat[j, :], kernel=kernel, sigma=sigma)
                Kii = kernel_fun(data_mat[i, :], data_mat[i, :], kernel=kernel, sigma=sigma)
                Kjj = kernel_fun(data_mat[j, :], data_mat[j, :], kernel=kernel, sigma=sigma)
                eta = 2.0 * Kij - Kii - Kjj
                # 这里跟公式正好差了一个负号，对应公式里的 K11+K22-2*K12 <=0，即开口向下，或为0成一条直线的情况不考虑
                """
                若核函数不满足Mercer定理，那么会存在K_{11}+K_{22}-2K_{12}=0或者K_{11}+K_{22}-2K_{12}<0，
                当表达式为0的时候，是条直线，最小值就是边界值，算算两个边界值，哪个小就用哪个；
                假如表达式大于0，是个开口向下的二次函数，最小值也在边界取得。
                """
                if eta >= 0:
                    continue
                alphas[j] = (alphas[j] - label_mat[j] * (Ei - Ej)) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                # 如果更新范围很小说明不需要再更新
                if abs(alphas[j] - alpha_j_old) < 0.0001:
                    continue

                alphas[i] = alphas[i] + label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

                # 第四小段代码

                b1 = b - Ei + label_mat[i] * (alpha_i_old - alphas[i]) * Kii + \
                     label_mat[j] * (alpha_j_old - alphas[j]) * Kij
                b2 = b - Ej + label_mat[j] * (alpha_j_old - alphas[j]) * Kjj + \
                     label_mat[i] * (alpha_i_old - alphas[i]) * Kij
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
        if alpha_pairs_changed == 0:
            n_iter += 1
        else:
            n_iter = 0
        if count == 1000:
            print(alpha_pairs_changed)
            count = 0
    return b, alphas


def loadDataSet(filename):  # 加载数据
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))  # 一维列表
    return np.array(dataMat), np.array(labelMat).reshape(len(labelMat), 1)


def test_draw():
    path = 'E:\\机器学习\\实验报告\\实验4\\'
    file_name = r"E:\PythonProject\Machine learning\SVM\data.txt"
    # file = pd.read_table(file_name, header=None, names=["factor1", "factor2", "class"])
    # file.head()

    # positive = file[file["class"] == 1]
    # negative = file[file["class"] == -1]
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.scatter(positive["factor1"], positive["factor2"], s=30, c="b", marker="o", label="class 1")
    # ax.scatter(negative["factor1"], negative["factor2"], s=30, c="r", marker="x", label="class -1")
    # ax.legend()
    # ax.set_xlabel("factor1")
    # ax.set_ylabel("factor2")
    data_mat, label_mat = loadDataSet(file_name)
    b, alphas = smo_simple(data_mat, label_mat, C=0.6,
                           toler=0.001, max_iter=10, kernel='linear', sigma=1.0)
    print(b, alphas[alphas > 0])
    print(np.sum(alphas > 0))

    support_x = []
    support_y = []
    class1_x = []
    class1_y = []
    class01_x = []
    class01_y = []
    for i in range(alphas.shape[0]):
        if alphas[i] > 0.0:
            support_x.append(data_mat[i, 0])
            support_y.append(data_mat[i, 1])
    for i in range(alphas.shape[0]):
        if label_mat[i] == 1:
            class1_x.append(data_mat[i, 0])
            class1_y.append(data_mat[i, 1])
        else:
            class01_x.append(data_mat[i, 0])
            class01_y.append(data_mat[i, 1])
    print('support vector:')
    for (x, y) in zip(support_x, support_y):
        print((x, y))
    w_best = np.dot(np.multiply(alphas, label_mat).T, data_mat)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(support_x, support_y, s=100, c="y", marker="v", label="support_v")
    ax.scatter(class1_x, class1_y, s=30, c="b", marker="o", label="class 1")
    ax.scatter(class01_x, class01_y, s=30, c="r", marker="x", label="class -1")
    lin_x = np.linspace(-3, 4)
    lin_y = (-float(b) - w_best[0, 0] * lin_x) / w_best[0, 1]
    plt.plot(lin_x, lin_y, color="black")
    ax.legend()
    ax.set_xlabel("factor1")
    ax.set_ylabel("factor2")
    plt.savefig(path + 'pointsSample.png')
    plt.show()


def getTest_f_xi(data_mat, label_mat, test_data, b, alphas, kernel_type, sigma):
    KXXi = kernel_fun(data_mat, test_data, kernel=kernel_type, sigma=sigma)
    f_xi = np.multiply(alphas, label_mat).T @ KXXi + b
    # test_result = np.zeros((test_data.shape[0], 1))
    # for i in range(f_xi.size):
    #     if f_xi >= 1:
    #         test_result[i] = 1
    #     else:
    #         test_result[i] = -1
    # return test_result
    return f_xi


def classicalSVM(n_class, train_data, train_data_index, test_data, test_data_index,
                 C=0.6, toler=0.001, max_iter=10, kernel_type='linear', sigma=1.0):
    b_list = []
    alphas_list = []
    label_mat_list = []
    # 对每个人训练一个与其他类分开的的超平面
    for j in range(n_class):

        label_mat = np.zeros(shape=train_data_index.shape)
        for k in range(train_data_index.shape[0]):
            if train_data_index[k] == j:
                label_mat[k] = 1
            else:
                label_mat[k] = -1
        label_mat_list.append(label_mat)
        print("class %d " % (j + 1))
        # if j == 1:
        #     continue
        b, alphas = smo_simple(train_data, label_mat, C=C,
                               toler=toler, max_iter=max_iter, kernel=kernel_type, sigma=sigma)
        b_list.append(b)
        alphas_list.append(alphas)
        print('svm %d' % (j + 1))
        # print(b, alphas[alphas > 0])
        print('num of alphas')
        print(np.sum(alphas > 0))
    # Classify~~
    test_f_xi = np.zeros((test_data.shape[0], n_class))  # (n_test, n_class)
    for j in range(n_class):
        test_f_xi[:, j] = getTest_f_xi(train_data, label_mat_list[j], test_data,
                                       b_list[j], alphas_list[j], kernel_type, sigma)
    testResult = np.argmax(test_f_xi, axis=1).reshape(-1, 1)  # 找出最大f_xi的下标，作为它的类别
    return np.mean(test_data_index == testResult)


def SVM_test(k=5, fun_name='classical', data_name='orl',
             C=1.0, toler=0.001, max_iter=10, kernel_type='linear',
             sigma=1.0, n_used_class=None, PCA_dimension=None, LDA_dimension=None, LDA_mat=None):
    print('In SVM :\n' + 'SVM type: ' + fun_name)
    print('Kernel type: ' + kernel_type)
    print('In ' + data_name)
    print('K = %d' % k)

    # 加载数据
    data, n_class, n_one_class = get_data(data_name)

    # data = get_normalize(data)

    if PCA_dimension is not None:
        # PCA处理
        print('------------------')
        print('in PCA dimension = %d' % PCA_dimension)
        print('------------------')
        data = pcaProject(data, dimension=PCA_dimension)
        print('shape of data processed by PCA')

    elif LDA_dimension is not None:
        # LDA处理
        data = LDAProject(data, n_class, n_one_class, fun_name='regularized', dimension=LDA_dimension, LDA_mat=LDA_mat)
        print('-----------------------------------')
        print('in LDA dimension = %d' % data.shape[1])
        print('-----------------------------------')
        print('shape of data processed by LDA')

    if n_used_class is not None and n_used_class < n_class:
        # 选取一部分
        n_class = n_used_class
        data = data[:n_class * n_one_class]

    data = data / np.max(data)
    print(data.shape)
    accuracy = 0
    n_train = int(n_one_class * k / 10)
    if n_train == 1:
        n_train = 2
    print('n of train set: %d' % n_train)
    for i in range(n_one_class):
        # 处理数据集
        train_data, train_data_index, test_data, test_data_index = separateData(data, n_train, i, n_class)
        print(train_data.shape)
        # 标准化可以提高识别率......吗？
        # train_data = get_normalize(train_data)
        # test_data = get_normalize(test_data)

        # PCA和LDA加工区

        # End PCA and LDA

        # SVM start
        res = classicalSVM(n_class, train_data, train_data_index, test_data, test_data_index,
                           C=C, toler=toler, max_iter=max_iter, kernel_type=kernel_type, sigma=sigma)

        print('---------------------------------------\n')
        print("res %d" % (i + 1))
        print(res)
        print('\n---------------------------------------')
        accuracy += res

    accuracy /= n_one_class
    print('---------------------------------------')
    print('---------------------------------------')
    print('---------------------------------------')
    print('In ' + fun_name + ' accuracy: ')
    print(accuracy)
    print('---------------------------------------')
    print('---------------------------------------')
    print('---------------------------------------')
    return accuracy


def SVM_figure_d(data_name):
    path = 'E:\\机器学习\\实验报告\\实验4\\'
    f = open(path + data_name + "_svm_dif_d.txt", "w+")
    print('--------------------------------------------------')
    print('For different d:')
    print('--------------------------------------------------')

    # 调参
    PCA_C = 0
    PCA_sigma = 0
    LDA_C = 0
    LDA_sigma = 0
    LDA_n_class = 0
    k = 5  # 以下参数只适用于K=5！！
    if data_name == 'yale':
        PCA_C = 2.5
        PCA_sigma = 0.55
        LDA_C = 1
        LDA_sigma = 0.55
        LDA_n_class = 15
    elif data_name == 'orl':
        PCA_C = 0.5
        PCA_sigma = 0.55
        LDA_C = 1
        LDA_sigma = 0.55
        LDA_n_class = 40
    elif data_name == 'feret':
        PCA_C = 0.5
        PCA_sigma = 0.55
        LDA_C = 0.5
        LDA_sigma = 0.55
        LDA_n_class = 200
    elif data_name == 'ar':
        PCA_C = 0.5
        PCA_sigma = 0.55
        LDA_C = 0.4
        LDA_sigma = 0.55
        LDA_n_class = 120

    name_list = ['PCA processed', 'LDA processed']
    f.write('d from 10 to 160, step 10:\n')
    pca_x = [i for i in range(10, 170, 10)]
    lda_x = [i for i in range(9, LDA_n_class, (LDA_n_class - 9) // 10)]
    y = []
    for i in range(len(name_list)):
        plt.figure(i)
        if name_list[i] == 'PCA processed':
            y.append([SVM_test(k=k, fun_name='classical', data_name=data_name,
                               C=PCA_C, toler=0.1, max_iter=3, kernel_type='rbf',
                               sigma=PCA_sigma, n_used_class=10, PCA_dimension=pca_x[j])
                      for j in range(len(pca_x))])
            plt.plot(pca_x, y[i], label=name_list[i], c=colors[i])
        elif name_list[i] == 'LDA processed':
            data, n_class, n_one_class = get_data(data_name)
            LDA_mat = LDATrain(data, n_class, n_one_class, fun_name='regularized')
            y.append([SVM_test(k=k, fun_name='classical', data_name=data_name,
                               C=LDA_C, toler=0.1, max_iter=3, kernel_type='rbf',
                               sigma=LDA_sigma, n_used_class=10, LDA_dimension=lda_x[j], LDA_mat=LDA_mat)
                      for j in range(len(lda_x))])
            plt.plot(lda_x, y[i], label=name_list[i], c=colors[i])
        f.write(name_list[i] + ':\n' + str(y[i]) + '\n')
        plt.title(data_name + u"人脸库")
        plt.xlabel("dimension")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(path + data_name + name_list[i] + '.png')
        plt.show()
    f.close()


if __name__ == '__main__':
    # test_draw()
    # yale:c=10,sigma=0.01
    # yale k=2, pca=10, res=0.79 ----------too slow

    # yale:c=1,sigma=0.5
    # yale k=2, LDA=10, res=0.97 ----------good
    # SVM_test(k=5, fun_name='classical', data_name='feret',
    #          C=0.5, toler=0.1, max_iter=3, kernel_type='rbf', sigma=0.55,
    #          n_used_class=10, PCA_dimension=None, LDA_dimension=160)

    # SVM_test(k=5, fun_name='classical', data_name='ar',
    #          C=0.5, toler=0.1, max_iter=3, kernel_type='rbf', sigma=0.55,
    #          n_used_class=10, PCA_dimension=None, LDA_dimension=160)
    # SVM_test(k=5, fun_name='classical', data_name='ar',
    #          C=0.5, toler=0.1, max_iter=3, kernel_type='rbf', sigma=0.55,
    #          n_used_class=10, PCA_dimension=100, LDA_dimension=None)

    # SVM_test(k=5, fun_name='classical', data_name='iris',
    #      C=10, toler=0.001, max_iter=3, kernel_type='rbf', sigma=0.1)
    # iris = datasets.load_iris()
    # print(iris)
    # SVM_figure_d('ar')
    SVM_figure_d('feret')
