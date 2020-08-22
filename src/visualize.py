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


def plot_feature_map():
    from model import CNN3
    model = CNN3()
    model.load_weights('../models/cnn3_best_weights.h5')

    def get_feature_map(model, layer_index, channels, input_img):
        from tensorflow.keras import backend as K
        layer = K.function([model.layers[0].input], [model.layers[layer_index].output])
        feature_map = layer([input_img])[0]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8))
        for i in range(channels):
            img = feature_map[:, :, :, i]
            plt.subplot(4, 8, i + 1)
            plt.imshow(img[0], cmap='gray')
        plt.savefig('rst.png')
        plt.show()
        import cv2
        img = cv2.cvtColor(cv2.imread('../data/demo.jpg'), cv2.cv2.COLOR_BGR2GRAY)
        img.shape = (1, 48, 48, 1)
        get_feature_map(model, 4, 32, img)


if __name__ == '__main__':
    import numpy as np
    history = load_file('../train_results/his_cnn2.pkl')
    print(np.max(history['val_acc']))
    plot_loss(history)
    plot_acc(history)
