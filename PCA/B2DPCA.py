import numpy as np


def twoDPCA(imgs, k):
    a, b, c = imgs.shape
    average = np.zeros((b, c))
    for i in range(a):
        average += imgs[i, :, :] / (a * 1.0)  #
    G_t = np.zeros((c, c))
    for j in range(a):
        img = imgs[j, :, :]
        temp = img - average
        G_t = G_t + np.dot(temp.T, temp) / (a * 1.0)
    eig_value, vec = np.linalg.eigh(G_t)
    index = np.argsort(-eig_value)  # 对特征值大到小排序, 返回索引列表
    return vec[:, index[:k]]  # (c, k)
    # eig_value_sum = sum(eig_value)
    # for k in range(c):
    #     alpha = sum(eig_value[:k]) * 1.0 / eig_value_sum
    #     if alpha >= p:
    #         return vec[:, :k]


def BTwoDPCA(imgs, k):
    u = twoDPCA(imgs, k)
    a1, b1, c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i, :, :], u)  # (b, c) @ (c, k) = (b, k)
        img.append(temp1.T)  # m (k, b)
    img = np.array(img)  # (m, k, b)
    uu = twoDPCA(img, k)  # (b, k)
    return u, uu
