import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔


colors = ['r', 'g', 'b', 'm', 'k']  # 红，绿，蓝，酒红，黑
markers = ['o', 'x', '+', '^', 'v']  # 圆圈， x, +, 正三角， 倒三角
styles = [x + y for x in colors for y in markers]  # 这样就有5*5=25中不同组合


def drawFigure(data_set, W):
    plt.figure(1)
    # plt.subplot(221)
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_position(("data", 0))
    i = 0
    for data in data_set:
        plt.scatter(data[:, 0].reshape(-1).tolist(), data[:, 1].reshape(-1).tolist(), c=colors[i % 5],
                    marker=markers[i // 5 % 5])  # 5是颜色或形状的个数
        i += 1
    plt.xlabel("x")
    plt.ylabel("y")

    # 设置间隔示例
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.xlabel('Numbers', fontsize=14)
    # plt.ylabel('Squares', fontsize=14)
    # x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    # ax = plt.gca()
    # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    # plt.xlim(-0.5, 11)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-5, 110)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    # print(matplotlib.matplotlib_fname())
    # 字体

    # 画出W对应的直线
    x = np.arange(-20, 20, 1)
    y = W[1, 0] / W[0, 0] * x
    plt.plot(x, y, c="b")
    plt.show()

    # 先展示二维
