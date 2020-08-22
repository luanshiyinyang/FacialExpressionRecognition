"""
author: Zhou Chen
datetime: 2019/6/26 14:26
desc: 绘制小论文部分图像
"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_grad(x):
    return 1 - tanh(x) ** 2


def relu(x):
    return np.where(x <= 0, 0, x)


def relu_grad(x):
    x = np.where(x <= 0, 0, x)
    x = np.where(x > 0, 1, x)
    return x


def plot_data(f, f_grad):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    # 去掉边框
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # 移位置 设为原点相交
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    x = np.linspace(-10, 10, 1000)
    y = f(x)
    plt.plot(x, y)
    plt.title("relu")

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    # 去掉边框
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    # 移位置 设为原点相交
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    x = np.linspace(-10, 10, 1000)
    y = f_grad(x)
    plt.plot(x, y)
    plt.title("relu grad")
    plt.savefig("relu.png")
    plt.show()


if __name__ == '__main__':
    # plot_data(sigmoid, sigmoid_grad)
    # plot_data(tanh, tanh_grad)
    plot_data(relu, relu_grad)