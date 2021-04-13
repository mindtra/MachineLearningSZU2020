import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = ['lime', 'r', 'b', 'm', 'slategrey', 'c', 'y', 'k', 'sandybrown', 'g']
markers = ['o', 'x', '+', '^', 'v', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
           'P', 'X']  # 圆圈， x, +, 正三角， 倒三角...
styles = [x + y for x in colors for y in markers]  # 这样就有5*5=25中不同组合
matplotlib.rcParams['axes.unicode_minus'] = False  # 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签


def draw_fig(data_name='ar'):
    from PCA.PCATest import pca_test
    from LDA.LDAtest1 import LDA_test
    from SVM.svm import SVM_test
    print('--------------------------------------------------')
    print('For different K:')
    print('--------------------------------------------------')
    # name_list = ['linear', 'Locality', 'KRRC', 'USSL', 'rbf']
    # name_list = ['linear', 'Locality', 'KRRC', 'rbf']
    # name_list = ['linear', 'regularized', 'pca']
    name_list = ['svm LDA processed', 'svm PCA processed', 'regularized', 'pca']
    # num = 10  # 每个人的张数
    plt.figure(1)
    x = [i for i in range(2, 10)]  # k的个数
    if data_name == 'ar':
        svmC = [0.5, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2]
    else:
        svmC = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4]
    y = []

    # 参数

    path = 'E:\\机器学习\\实验报告\\实验4\\'

    f = open(path + data_name + "_dif_way.txt", "w+")
    # f = open(path + data_name + "_dif_regressionWithoutUSSL.txt", "w+")
    # f = open(path + data_name + "_Different_K&way.txt", "w+")
    f.write('K from 2 to 9:\n')
    for i in range(len(name_list)):
        if name_list[i] == 'pca':
            y.append([pca_test(x[j], data_name=data_name, n_used_class=10, dimension=120) for j in range(len(x))])
        elif name_list[i] == 'regularized':
            y.append([LDA_test(x[j], fun_type='regularized', data_name=data_name, n_used_class=10)
                      for j in range(len(x))])
        # elif name_list[i] == 'rbf':
        #     y.append([linear_test(x[j], fun_name='KRRC', data_name=data_name, kernel_type='rbf')
        #               for j in range(len(x))])
        # else:
        #     y.append([linear_test(x[j], fun_name=name_list[i], data_name=data_name, kernel_type='poly')
        #               for j in range(len(x))])
        elif name_list[i] == 'svm LDA processed':
            y.append([SVM_test(k=x[j], fun_name='classical', data_name=data_name,
                               C=svmC[j], toler=0.1, max_iter=3, kernel_type='rbf',
                               sigma=0.55, n_used_class=10, PCA_dimension=None, LDA_dimension=999)
                      for j in range(len(x))])
        elif name_list[i] == 'svm PCA processed':
            y.append([SVM_test(k=x[j], fun_name='classical', data_name=data_name,
                               C=svmC[j], toler=0.1, max_iter=3, kernel_type='rbf',
                               sigma=0.55, n_used_class=10, PCA_dimension=120, LDA_dimension=None)
                      for j in range(len(x))])

        print("k of max accuracy in " + name_list[i] + ": %d" % x[int(np.argmax(y[i]))])
        plt.plot(x, y[i], label=name_list[i], c=colors[i])
        f.write(name_list[i] + ':\n' + str(y[i]) + '\n')
    plt.title(data_name + u"人脸库")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(path + data_name + '_dif_way.png')
    plt.show()
    f.close()


if __name__ == '__main__':
    draw_fig(data_name='feret')
    draw_fig(data_name='ar')
    os.system('shutdown -s -f -t 60')
