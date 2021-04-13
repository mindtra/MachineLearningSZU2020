import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import copy

from LDA.LDAtest1 import KNN, get_normalize
from PCA.PCATest import pcaKNN
from PCA.pca import mean_of_matrix, pca
from dataprocess import get_data, separateData
from kernel import kernel_fun

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
colors = ['r', 'g', 'b', 'm', 'k', 'c', 'y', 'slategrey', 'sandybrown', 'lime']
matplotlib.rcParams['axes.unicode_minus'] = False  # 设置字体


class Regression:

    def __init__(self):
        pass

    def train_gradient_descent(self, X, y, learning_rate=0.01, n_iters=100, type='linear', n_class=1, lam=0.01,
                               n_batch=None):
        """
        梯度下降法
        :param y: 输出
        :param learning_rate: 学习率
        :param n_iters: 迭代次数
        :param type:
        :param n_class:
        :param lam:
        :param n_batch:
        :return: self.weights, costs
        """
        X = self.convert_X(X)
        n_features = X.shape[1]
        y_one_hot = 0
        if n_batch is None:
            n_batch = X.shape[0]
        if type == 'softmax':
            self.weights = np.zeros(shape=(n_features, n_class))
            y_one_hot = one_hot(y)
            self.bias = 0
        elif type == 'linear':
            self.weights = np.zeros(shape=(n_features, y.shape[1]))
            self.bias = 0

        costs = []
        for i in range(n_iters):
            # i = 0
            # cost = 1000
            # while cost >= 0.5:

            # 计算得分
            rand_index = np.random.choice(range(X.shape[0]), n_batch, replace=False)
            scores = np.dot(X[rand_index], self.weights) + self.bias

            # 损失函数
            cost = 0
            probs = 0
            if type == 'linear':
                cost = self.get_linear_cost(scores, y[rand_index])
            elif type == 'softmax':
                # softmax计算概率
                probs = self.softmax(scores)  # m * n_class
                cost = -(1.0 / n_batch) * np.sum(y_one_hot[rand_index] * np.log(probs))
            costs.append(cost)

            if i % 100 == 0:
                print(f"Cost at iteration {i}: {cost}")

            # 计算梯度
            dJ_dw = 0
            if type == 'linear':
                dJ_dw = (2 / n_batch) * np.dot(X[rand_index].T, (scores - y[rand_index]))
            elif type == 'softmax':
                dJ_dw = -(1 / n_batch) * np.dot(X[rand_index].T, (y_one_hot[rand_index] - probs)) + lam * self.weights
                dJ_dw[0, :] = dJ_dw[0, :] - lam * self.weights[0, :]
            # dJ_db = (2 / n_samples) * np.sum((scores - y))

            # 检测梯度防止爆炸
            if type == 'linear':
                clip_gradient = 10000
                dw_norm = np.linalg.norm(dJ_dw)
                for j in range(n_features):
                    if dw_norm > clip_gradient:
                        dJ_dw[j] = dJ_dw[j] * (clip_gradient / dw_norm)

            # 计算权值
            self.weights = self.weights - learning_rate * dJ_dw
            # self.bias = self.bias - learning_rate * dJ_db  # 在输入X前将矩阵加一列1可以不用求此步

            # i += 1  # in while
        return self.weights, costs

    def _get_batches(self, X, y, n_batch):
        start = 0
        end = 0
        X_batches = []
        y_batches = []
        for i in range(X.shape[0]):
            end += 1
            if end != 0 and end % n_batch == 0:
                X_batch = X[start:end]
                y_batch = y[start:end]
                start = end
                X_batches.append(np.array(X_batch))
                y_batches.append(np.array(y_batch))
        return X_batches, y_batches

    def get_linear_cost(self, y_predict, y):
        return (1 / y_predict.shape[0]) * np.sum((y_predict - y) ** 2)

    def softmax(self, X):
        # 为了避免求exp(x)出现溢出的情况，一般需要减去最大值。\
        # 计算每行的最大值
        row_max = X.max(axis=1)

        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max = row_max.reshape(-1, 1)
        X = X - row_max

        # 计算e的指数次幂
        x_exp = np.exp(X)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        res = x_exp / x_sum
        return res

    def train_normal_equation(self, X, label, k, n_man, type=None, lam=0.01):
        """
        正规方程法
        """
        # 下面是普通的正规方程, 此处为按行放
        # X = self.convert_X(X)
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        # self.bias = 0
        # return self.weights, self.bias

        # 在人脸识别中，用回归的形式使重构误差最小，此时y不是训练集的标签而是测试集
        # 并且此处的数据变为按列放！！
        # self.y_data = copy.deepcopy(y)
        self.k = k
        self.n_man = n_man
        X = self.convert_X(X)
        X = X.T
        self.H = []  # 每个类别的投影矩阵
        self.label = np.zeros(n_man)
        for i in range(n_man):
            if type is None:
                xTx = X[:, i * k:(i + 1) * k].T @ X[:, i * k:(i + 1) * k]
                self.H.append(X[:, i * k:(i + 1) * k] @ np.linalg.pinv(xTx) @ X[:, i * k:(i + 1) * k].T)
            elif type == 'ridge':
                eye_mat = np.eye(k)  # 生成单位阵
                xTx = X[:, i * k:(i + 1) * k].T @ X[:, i * k:(i + 1) * k] + lam * eye_mat
                self.H.append(X[:, i * k:(i + 1) * k] @ np.linalg.inv(xTx) @ X[:, i * k:(i + 1) * k].T)
            self.label[i] = label[i * k]
            # print(self.H[i].shape)
        # eig_value, vec = np.linalg.eig(temp)
        # eig_index = np.argsort(-eig_value)
        # e = np.linalg.matrix_rank(temp)
        #
        # for i in range(e):
        #     temp_pinv = vec[:, eig_index[i]].T

    def normal_equation_predict(self, test_data):
        n_test = test_data.shape[0]
        test_data = self.convert_X(test_data)
        test_data = test_data.T
        error = np.zeros((self.n_man, n_test))
        for i in range(self.n_man):
            # 求每个测试集在每类投影矩阵下的重构误差
            error[i] = np.linalg.norm(test_data - self.H[i] @ test_data, axis=0)
        index = np.argmin(error, axis=0)  # 找到最小的位置
        return self.label[index].reshape(-1, 1)  # 返回对应的标签

    def linear_predict(self, X):
        X = self.convert_X(X)
        return np.dot(X, self.weights) + self.bias

    def softmax_predict(self, X):
        X = self.convert_X(X)
        scores = np.dot(X, self.weights) + self.bias
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1).reshape(-1, 1)

    def convert_X(self, X):
        return np.c_[np.ones((X.shape[0])), X]


def one_hot(y):
    n_class = len(np.unique(y.tolist()))  # 判断y中有几个分类
    eyes_mat = np.eye(n_class)  # 按分类数生成对角线为1的单位阵
    y_one_hot = np.zeros((y.shape[0], n_class))  # 初始化y的one_hot编码矩阵
    for i in range(0, y.shape[0]):
        y_one_hot[i] = eyes_mat[int(y[i])]  # 根据每行y值，更新one_hot编码矩阵
    # y_one_hot[np.arange(y.shape[0]), y.T] = 1  # 这行代码y乱写就越界了，这里仅供参考
    return y_one_hot


def get_graph_matrix(y_label, type='lda'):
    """
    获得图嵌入框架
    :param y_label:
    :param type:
    :return:
    """
    m = y_label.shape[0]
    W = np.zeros((m, m))
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if y_label[i] == y_label[j]:
                if type.lower() == 'lda':
                    W[i, j] = W[j, i] = 1
    for i in range(m):
        for j in range(m):
            if W[i, j] == 1:
                D[i, i] += 1
    return W, D


class KRRC:
    def __init__(self, X, label, n_class, lam=0.005, sigma=1.0, kernel='linear', locality=False):
        """
        kernel: 可选的核函数，默认为'linear'
            'linear'==>线性核函数；'poly'==>多项式核函数；'rbf'==>高斯核
        :param X:
        :param label:
        :param n_class:
        :param lam:岭回归的 λ 参数
        :param sigma:控制核函数的sigma
        :param kernel:
        """
        self.X = np.c_[np.ones((X.shape[0])), X]
        self.n_sample, self.X_feature = self.X.shape
        self.label_feature = label.shape[1]
        self.n_class = n_class
        self.batch = self.n_sample // self.n_class
        if self.label_feature == 1:
            self.label = label.reshape(-1, 1)
        else:
            self.label = label
        self.lam = lam
        self.sigma = sigma
        self.kernel = kernel
        self.batch = self.n_sample // self.n_class
        self.locality = locality

    def __get_weight(self, test_point):
        """
        :param test_point:shape should be (1, n_features)
        :return:weight,shape=(n_sample, n_sample)
        """
        weight = np.eye(self.n_sample)  # 初始化权重矩阵

        # 确定权重
        for i in range(self.n_sample):
            diffMat = test_point - self.X[i, :]
            weight[i, i] = np.exp((diffMat @ diffMat.T) / (-2 * self.sigma ** 2))

        return weight

    def __get_A(self, test_arr):
        """

        :param SVM_test:shape should be (1,n_features)
        :return:A len:n_class; A[i] shape: (self.batch*1)
        """
        test_arr = test_arr.reshape(1, -1)
        A = []
        eye_mat = np.eye(self.batch)  # 生成单位阵
        weight = 0
        if self.locality is True:
            weight = self.__get_weight(test_arr)
        for i in range(self.n_class):
            Xi = self.X[i * self.batch:(i + 1) * self.batch]
            K_Xx = kernel_fun(Xi, test_arr, self.kernel, self.sigma)  # 计算训练集和测试集核矩阵
            if self.locality is True:
                temp_w = weight[i * self.batch:(i + 1) * self.batch, i * self.batch:(i + 1) * self.batch]
                A.append(np.linalg.solve(temp_w @ self.K_XX[i] + self.lam * eye_mat, temp_w @ K_Xx))
            else:
                A.append(np.linalg.solve(self.K_XX[i] + self.lam * eye_mat, K_Xx))
        return A

    def classify(self, test):
        # 生成每一类对应的核矩阵，即K(Xi,Xi)
        test = np.c_[np.ones((test.shape[0])), test]
        self.K_XX = []
        eye_mat = np.eye(self.batch)  # 生成单位阵
        res = np.zeros(self.n_class)
        res_label = np.zeros(test.shape[0])
        for i in range(self.n_class):
            Xi = self.X[i * self.batch:(i + 1) * self.batch]
            self.K_XX.append(kernel_fun(Xi, Xi, self.kernel, self.sigma))
        for i in range(test.shape[0]):  # 对每个测试集
            A = self.__get_A(test[i].reshape(1, -1))
            for j in range(len(A)):
                res[j] = A[j].reshape(1, -1) @ (self.K_XX[j] + 2 * self.lam * eye_mat) @ A[j].reshape(-1, 1)
            res_label[i] = np.argmax(res)  # 找到最大值对应的标签
        return res_label.reshape(-1, 1)


class LocalityRegression:
    def __init__(self, X, label, n_class, lam=0.005, sigma=1.0):
        """
        :param X:shape should be (n_samples, n_features)
        :param label:shape should be (n_samples, n_features) or (n_samples, 1)
        :param sigma:控制权值计算的参数，太小过拟合，太大欠拟合
        """
        self.X = np.c_[np.ones((X.shape[0])), X]
        self.n_sample, self.X_feature = self.X.shape
        self.label_feature = label.shape[1]
        self.n_class = n_class
        self.batch = self.n_sample // self.n_class
        if self.label_feature == 1:
            self.label = label.reshape(-1, 1)
        else:
            self.label = label
        self.k = sigma
        self.lam = lam
        self.__get_XX()

    def __get_weight(self, test_point):
        """
        :param test_point:shape should be (1, n_features)
        :return:weight,shape=(n_sample, n_sample)
        """
        weight = np.eye(self.n_sample)  # 初始化权重矩阵

        # 确定权重
        for i in range(self.n_sample):
            diffMat = test_point - self.X[i, :]
            weight[i, i] = np.exp((diffMat @ diffMat.T) / (-2 * self.k ** 2))

        return weight

    def __get_hat(self, weight, test_arr):

        I = np.eye(self.batch)  # batch为每一类的个数
        W = []
        for i in range(self.n_class):
            temp_w = weight[i * self.batch:(i + 1) * self.batch, i * self.batch:(i + 1) * self.batch]
            temp_X = self.X[i * self.batch:(i + 1) * self.batch]
            wxxT = temp_w @ self.XXT[i] + self.lam * I
            A = np.linalg.solve(wxxT, temp_w @ temp_X @ test_arr.reshape(-1, 1))
            W.append(temp_X.T @ A)  # n * n
            # theta.append(temp_X.T @ np.linalg.inv(wxxT) @ temp_w @ temp_X)  # n * n

        return W

    def __get_XX(self):
        self.XXT = []
        for i in range(self.n_class):
            temp_X = self.X[i * self.batch:(i + 1) * self.batch]
            self.XXT.append(temp_X @ temp_X.T)

    def predict(self, X_test):
        """
        预测用标签，所有数据按行放
        :param X_test:
        :return:
        """
        n_test = X_test.shape[0]
        X_test = np.c_[np.ones((X_test.shape[0])), X_test]
        res = np.zeros(shape=self.label.shape)
        # 获得单位阵
        I = np.eye(self.X_feature)
        for i in range(n_test):
            weight = self.__get_weight(X_test[i])
            xTwx = self.X.T @ weight @ self.X
            theta = np.linalg.inv(xTwx + self.lam * I) @ self.X.T @ weight
            res[i] = X_test[i] @ theta @ self.label
        return res

    def classify(self, X_test, type='one_hot'):
        if type == 'one_hot':
            return self.__classify_one_hot(X_test)
        elif type == 'hat':
            return self.__classify_hat(X_test)

    def __classify_one_hot(self, X_test):
        y_hot = one_hot(self.label)
        n_test = X_test.shape[0]
        X_test = np.c_[np.ones((X_test.shape[0])), X_test]
        res = np.zeros((X_test.shape[0], self.n_class))
        I = np.eye(self.X_feature)
        for i in range(n_test):
            weight = self.__get_weight(X_test[i])
            xTwx = self.X.T @ weight @ self.X
            theta = np.linalg.solve(xTwx + self.lam * I, self.X.T @ weight @ y_hot)
            res[i] = X_test[i] @ theta
        return np.argmax(res, axis=1).reshape(-1, 1)

    def __classify_hat(self, X_test):
        # 在分类任务中，用回归的形式使重构误差最小，此时y不是训练集的标签而是测试集
        # 并且此处的数据变为按列放！！
        n_test = X_test.shape[0]
        X_test = np.c_[np.ones((X_test.shape[0])), X_test]
        e = np.zeros((self.n_class, 1))
        res_index = np.zeros((X_test.shape[0], 1))
        for i in range(n_test):
            test_arr = X_test[i].reshape(1, -1)
            weight = self.__get_weight(X_test[i])
            H = self.__get_hat(weight, X_test[i])
            for j in range(self.n_class):
                e[j] = np.linalg.norm(test_arr.reshape(-1, 1) - H[j], axis=0)
            res_index[i] = self.label[self.batch * np.argmin(e, axis=0)]
        return res_index


class USSL:
    """
    数据均为(n_sample,n_features)
    """

    def __init__(self, X, y_label, type='lda'):
        # self.X = get_normalize(X)
        self.X = X
        self.y_label = y_label
        self.m = y_label.shape[0]
        # self.flag = 1
        self.flag = 2
        if self.flag == 1:
            self.elastic = ElasticNet(a=0.5, b=0.5)

        else:
            self.X = np.c_[np.ones((X.shape[0])), X]
            self.elastic = sklearn.linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000)
        self.type = type

    def get_y_eig(self):
        W, D = get_graph_matrix(self.y_label, type=self.type)
        temp = np.linalg.inv(D) @ W
        eig_value, eig_vec = np.linalg.eigh(temp)
        rank = np.linalg.matrix_rank(np.linalg.inv(D) @ W)
        index = np.argsort(-eig_value)
        eig_vec = eig_vec.T
        y = [eig_vec[index[i]] for i in range(rank - 1)]
        return np.array(y).T

    def ussl_fit(self, type='ussl'):
        """
        获得投影矩阵P
        :return:
        """
        if type.lower() == 'ussl':
            self.Y = self.get_y_eig()
            self.elastic.fit(self.X, self.Y)
        elif type.lower() == 'one_hot':
            self.Y = one_hot(self.y_label)
            self.elastic.fit(self.X, self.Y)

    def classify(self, X_test):
        """
        分类，返回标签
        :param X_test:
        :return:
        """
        Y_train = self.elastic.predict(self.X)
        if self.flag != 1:
            X_test = np.c_[np.ones((X_test.shape[0])), X_test]
        Y_test = self.elastic.predict(X_test)
        return KNN(Y_train, Y_test, self.y_label)

    def predict(self, X_test):
        """
        返回投影出来的USSL中的y
        :param X_test:
        :return:
        """
        return self.elastic.predict(X_test)


class ElasticNet:
    """
    弹性回归网
    """

    def __init__(self, learning_rate=0.01, n=1000, a=0.5, b=1.0):
        self.learing_rate = learning_rate
        self.n = n
        self.a = a
        self.b = b

    def coordinate(self):
        for i in range(self.n_feature + 1):
            a = (1 / self.m) * (self.X[:, i].T @ self.X[:, i].reshape(-1, 1)) + self.a * (1 - self.b)
            dw = np.zeros(shape=self.w.shape)
            dw[i] = self.w[i]
            b = (1 / self.m) * (self.X.T @ (self.Y - self.X @ (self.w - dw)))[i]
            for k in range(self.Y.shape[1]):
                if b[k] < -self.a * self.b:
                    self.w[i, k] = (b[k] + self.a * self.b) / a
                elif b[k] > self.a * self.b:
                    self.w[i, k] = (b[k] - self.a * self.b) / a
                else:
                    self.w[i, k] = 0

    def fit(self, X, Y):
        self.m, self.n_feature = X.shape
        if len(Y.shape) == 1:
            self.w = np.zeros((self.n_feature + 1, 1))
        else:
            self.w = np.zeros((self.n_feature + 1, Y.shape[1]))
        self.X = np.c_[np.ones((X.shape[0])), X]
        self.Y = Y
        cost1 = self.costfun()
        cost2 = 0
        for i in range(self.n):
            print('in i == %d' % i)
            print(cost1)
            if abs(cost1 - cost2) <= 0.001:
                break
            else:
                cost1 = self.costfun()
                self.coordinate()
                cost2 = self.costfun()
        return self.w

    def costfun(self):
        return 1 / (2 * self.m) * ((self.X @ self.w - self.Y) ** 2).sum() \
               + self.a * self.b * (np.abs(self.w)).sum() + (self.a * (1 - self.b)) / 2 * (self.w ** 2).sum()

    def predict(self, X_test):
        X_test = np.c_[np.ones((X_test.shape[0])), X_test]
        return X_test @ self.w


# X_train, X_test ,y_train, y_test= train_test_split(X, y,test_size=0.3, rando
# m_state = 20, shuffle=True)
def linear_test(k=5, fun_name='linear', data_name='orl', kernel_type='linear', n_used_class=None):
    """

    :param k:
    :param fun_name: linear, Softmax, USSL, Locality, KRRC
    :param data_name:
    :param kernel_type: rbf, poly, linear
    :return:
    """
    print('In linear regression:\n' + 'Regression type: ' + fun_name)
    print('In ' + data_name)
    print('K = %d' % k)
    data, n_class, n_one_class = get_data(data_name)
    if fun_name.lower() != 'ussl':
        data = data / np.max(data)
    if n_used_class is not None:
        # 选取一部分
        n_class = n_used_class
        data = data[:n_class * n_one_class]
    # data = get_normalize(data)
    # project_W = pca(data)
    # d = 160
    # project_W = project_W[:d]
    # data = data @ project_W
    # data = mat["Yale5040165"].reshape((2000, 165))
    accuracy = 0
    accuracy2 = 0
    n_iters = 2000
    for i in range(n_one_class):
        # 处理数据集
        train_data, train_data_index, test_data, test_data_index = separateData(data, k, i, n_class)
        #         # 归一化可以提高识别率
        # train_data = get_normalize(train_data)
        # test_data = get_normalize(test_data)
        res = 0
        # 线性回归
        if fun_name.lower() == 'linear':
            re1 = Regression()
            # 调用库看看效果
            # re1 = sklearn.linear_model.LinearRegression()
            # re1.fit(train_data, train_data_index)
            # predict1 = re1.normal_equation_predict(train_data)
            # predict1 = re1.linear_predict(train_data)
            # re1.train_normal_equation(train_data, train_data_index, k, n_class)
            re1.train_normal_equation(train_data, train_data_index, k, n_class, type='ridge', lam=0.01)
            predict1 = re1.normal_equation_predict(test_data)
            # predict1 = re1.linear_predict(test_data)
            res = np.mean(test_data_index == predict1)

        # softmax梯度下降
        elif fun_name.lower() == 'softmax':
            re1 = Regression()
            W, costs = re1.train_gradient_descent(train_data, train_data_index, learning_rate=0.1, n_iters=n_iters,
                                                  type='softmax', n_class=n_class, n_batch=None)
            res = np.mean(test_data_index == re1.softmax_predict(test_data))
        # # draw costs
        # fig = plt.figure()
        # x = [i for i in range(n_iters)]
        # plt.plot(x, costs, c='r')
        # plt.show()

        # USSL 谱回归
        elif fun_name.lower() == 'ussl':
            re1 = USSL(train_data, train_data_index)
            re1.ussl_fit(type='ussl')
            # re1.ussl_fit(type='one_hot')
            res = np.mean(test_data_index == re1.classify(test_data))

        # 局部加权回归
        elif fun_name.lower() == 'locality':
            re1 = LocalityRegression(train_data, train_data_index, n_class, lam=0.005, sigma=1)
            # res = np.mean(test_data_index == re1.classify(test_data, type='one_hot'))
            res = np.mean(test_data_index == re1.classify(test_data, type='hat'))

        # 核岭回归
        elif fun_name.lower() == 'krrc':
            # re1 = KRRC(train_data, train_data_index, n_class, lam=0.005, sigma=1, kernel='rbf')
            print('kernel type: ' + kernel_type)
            re1 = KRRC(train_data, train_data_index, n_class, lam=0.005, sigma=3, kernel=kernel_type)
            # re1 = KRRC(train_data, train_data_index, n_class, lam=0.005, sigma=1, kernel='linear')
            # re1 = KRRC(train_data, train_data_index, n_class, lam=0.005, sigma=1, kernel='linear', locality=True)
            res = np.mean(test_data_index == re1.classify(test_data))

        # res = np.mean(train_data_index == pcaKNN(train_data, predict1, train_data_index))
        # print("res %d" % (i + 1))
        # print(res)
        # accuracy += res
        # predict1 = re1.normal_equation_predict(test_data)
        # predict1 = re1.softmax_predict(test_data)
        # predict1 = re1.linear_predict(test_data)
        # predict1 = predict1.astype(int)
        # res = np.mean(test_data_index == predict1)
        # res = np.mean(test_data_index == pcaKNN(test_data, predict1, test_data_index))
        print("res %d" % (i + 1))
        print(res)
        accuracy += res

    accuracy /= n_one_class
    print('In ' + fun_name + ' accuracy: ')
    print(accuracy)
    return accuracy


def regression_point_test():
    np.random.seed(123)

    X = 2 * np.random.rand(500, 1)
    y = 5 + 3 * X + np.random.randn(500, 1)
    # fig = plt.figure(figsize=(8, 6))
    fig = plt.figure()
    plt.scatter(X, y)
    plt.title("Dataset")
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f'Shape X_train: {X_train.shape}')
    print(f'Shape y_train: {y_train.shape}')
    print(f'Shape X_test: {X_test.shape}')
    print(f'Shape y_test: {y_test.shape}')
    # 添加一列全为1的列
    X_b_train = np.c_[np.ones((X_train.shape[0])), X_train]
    X_b_test = np.c_[np.ones((X_test.shape[0])), X_test]

    print(X_b_test)
    print(X_b_test.shape)

    re1 = Regression()
    re1.train_gradient_descent(X_train, y_train, 0.01, 600)
    y_predict1 = re1.linear_predict(X_test)
    cost1 = re1.get_linear_cost(y_predict1, y_test)
    plt.scatter(X_test, y_predict1, c='c')
    # plt.show()

    re2 = Regression()
    re2.train_normal_equation(X_b_train, y_train)
    y_predict2 = re2.linear_predict(X_b_test)
    cost2 = re2.get_linear_cost(y_predict2, y_test)
    plt.scatter(X_test, y_predict2, c='y')
    plt.show()

    print('cost1:')
    print(cost1)
    print('cost2:')
    print(cost2)


def regression_draw_fig():
    from PCA.PCATest import pca_test
    from LDA.LDAtest1 import LDA_test
    print('--------------------------------------------------')
    print('For different K:')
    print('--------------------------------------------------')
    # name_list = ['linear', 'Locality', 'KRRC', 'USSL', 'rbf']
    name_list = ['linear', 'Locality', 'KRRC', 'rbf']
    # name_list = ['linear', 'regularized', 'pca']
    num = 10  # 每个人的张数
    plt.figure(3)
    x = [i for i in range(2, num)]  # k的个数
    y = []
    data_name = 'orl'
    # 改数据集记得改代码1,2,3
    path = 'E:\\机器学习\\实验报告\\实验3\\'
    # f = open(path + data_name + "_dif_regression1.txt", "w+")
    f = open(path + data_name + "_dif_regressionWithoutUSSL.txt", "w+")
    # f = open(path + data_name + "_Different_K&way.txt", "w+")
    f.write('K from 2 to 9:\n')
    for i in range(len(name_list)):
        if name_list[i] == 'pca':
            y.append([pca_test(x[j], data_name=data_name) for j in range(len(x))])
        elif name_list[i] == 'regularized':
            y.append([LDA_test(x[j], fun_type='regularized', data_name=data_name) for j in range(len(x))])
        elif name_list[i] == 'rbf':
            y.append([linear_test(x[j], fun_name='KRRC', data_name=data_name, kernel_type='rbf')
                      for j in range(len(x))])
        else:
            y.append([linear_test(x[j], fun_name=name_list[i], data_name=data_name, kernel_type='poly')
                      for j in range(len(x))])
        print("k of max accuracy in " + name_list[i] + ": %d" % x[int(np.argmax(y[i]))])
        plt.plot(x, y[i], label=name_list[i], c=colors[i])
        f.write(name_list[i] + ':\n' + str(y[i]) + '\n')
    plt.title(data_name + u"人脸库")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.legend()
    # plt.savefig(path + data_name + '_dif_regression1.png')
    plt.savefig(path + data_name + '_dif_regressionWithoutUSSL.png')
    # plt.savefig(path + data_name + '_Different_K&way.png')
    plt.show()
    f.close()


if __name__ == '__main__':
    # regression_point_test()
    linear_test(k=5, fun_name='linear', data_name='orl', kernel_type='linear', n_used_class=10)
    # regression_draw_fig()
    # e = np.eye(5)
    # e[0, 0] = 2
    # e[1, 1] = 3
    # e[3, 3] = -1
    # # e[np.arange(5), np.arange(5)] = 1 / e[np.arange(5), np.arange(5)]
    # print(e)
    # index = np.argmax(e, axis=0)
    # print(index)
    # print(e[index])
    #
    # label = [1, 3, 4, 5, 6]
    # label = np.array(label)
    # result = label[index]
    # print(result)
    # a = 8.64
    # b = 8.39
    # print((a - b) / b)
