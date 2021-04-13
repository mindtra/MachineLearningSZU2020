import numpy as np

# class Kernel:
#     '''
#     Input:
#         kernel: 可选的核函数，默认为'linear'
#             'linear'==>线性核函数；'poly'==>多项式核函数；'rbf'==>高斯核
#         n_components：The dimension after dimensionality reduction
#                 if n_components=0, n_components will be set by the refactoring threshold
#         t: threshold, t=0.95
#         sigma: the degree of rbf kernel function, polynomial kernel function
#     '''
#
#     def __init__(self, kernel='linear', n_components=0, t=0.95, sigma=1):
#         self.kernel = kernel
#         self.n_components = n_components
#         self.t = t
#         self.sigma = sigma
#         self.w = []  # [x_train num_sample, n_components]
#         self.K = []  # [x_train num_sample, x_train num_sample]
#         self.X = []  # [x_train num_sample, x_train num_feature]

'''
define rbf, linear and poly kernel function
Input:
    X1：numpy.ndarry,  size: [X1_num_sample, num_feature]
    X1：numpy.ndarry,  size: [X2_num_sample, num_feature]
Returns:
    k: numpy.ndarry,  size: [X2_num_sample, X1_num_sample]
'''


def rbf(X1, X2, sigma=1.0):
    m = X1.shape[0]
    p = X2.shape[0]
    ''' 
    使用pdist方法来求解欧式距离，运行速度较之于下面的for方法快，但...
    k = pdist(X1, 'euclidean')
    k = squareform(k)
    k = np.exp( -k/(2*self.sigma**2) ) 
    '''
    k = np.ones([m, p])
    for i in range(p):
        # dist = np.sqrt(np.sum(np.square(X1 - X2[i, :]), axis=1))
        dist = np.linalg.norm(X1 - X2[i, :], axis=1)
        k[:, i] = np.exp(-(dist**2 / (2 * sigma ** 2)))  # 这一行的写法与课本的表达式一致，等价于下面那种写法
        # k[i, :] = np.exp(-self.sigma * k)  # 这一行的写法是高斯核的另一种等价表达式
    return k


def linear(X1, X2):
    k = np.dot(X1, X2.T)
    return k


def poly(X1, X2, sigma=1.0):
    k = np.dot(X1, X2.T)
    k = k ** sigma
    return k


def kernel_fun(X1, X2, kernel='linear', sigma=1.0):
    if len(X1.shape) == 1:
        X1.reshape((1, X1.shape[0]))
    if len(X2.shape) == 1:
        X2.reshape((1, X2.shape[0]))
    if kernel.lower() == 'rbf':
        return rbf(X1, X2, sigma)
    elif kernel.lower() == 'poly':
        return poly(X1, X2, sigma)
    elif kernel.lower() == 'linear':
        return linear(X1, X2)
