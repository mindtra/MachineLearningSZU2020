import numpy as np


def creatData(size, num_of_C):  # size是数据的多少,num_of_C有几类
    data = [np.mat(np.random.random([size, 2]) * 5 + 2 + 13 * i) for i in range(0, num_of_C)]
    # s1 = np.mat(np.random.random([size, num_of_C]) * 5 + 15)
    # s2 = np.mat(np.random.random([size, num_of_C]) * 5 + 5)
    # for i in range(0, size):
    #     s1[i, 0] += 5
    return data
    # return data



