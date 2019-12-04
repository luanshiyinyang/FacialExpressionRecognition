# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/6/23 12:49
   desc: 可视化训练过程
"""


def load_file(filename):
    """

    :param filename:
    :return:
    """
    import pickle
    data_file = open(filename, 'rb')
    data = pickle.load(data_file)
    data_file.close()
    return data.history


def plot_loss(his):
    """

    :param his:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['loss'])), his['loss'], label='train loss')
    plt.plot(np.arange(len(his['val_loss'])), his['val_loss'], label='valid loss')
    plt.title('loss')
    plt.legend(loc='best')
    plt.savefig('../train_results/loss_cnn2.png')
    plt.show()


def plot_acc(his):
    """

    :param his:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['acc'])), his['acc'], label='train accuracy')
    plt.plot(np.arange(len(his['val_acc'])), his['val_acc'], label='valid accuracy')
    plt.title('accuracy')
    plt.legend(loc='best')
    plt.savefig('../train_results/accuracy_cnn2.png')
    plt.show()


if __name__ == '__main__':
    import numpy as np
    history = load_file('../train_results/his_cnn2.pkl')
    print(np.max(history['val_acc']))
    plot_loss(history)
    plot_acc(history)
