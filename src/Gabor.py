"""
author: Zhou Chen
datetime: 2019/7/1 16:54
desc: 利用Gabor滤波尝试实现
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os


class Decomposition(object):
    def mean_pooling(self, img, size):
        """
        :param img:
        :param size:
        :return:
        """
        n, m = img.shape
        fn, fm = int(n / size), int(m / size)
        fimg = np.zeros((fn, fm), dtype=float)
        for i in range(fn):
            for j in range(fm):
                sum = 0
                for x in range(i * size, i * size + size):
                    for y in range(j * size, j * size + size):
                        sum += img[x, y]
                fimg[i, j] = sum / (size * size)
        return fimg

    def PCA(self, train, test, n=1000):
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n)
        train = pca.fit_transform(train)
        test = pca.transform(test)
        return train, test


class Classifier(object):
    def knn(self, train_array, test_array):
        """
        :param train_array:
        :param test_array:
        :return:
        """

        # 存储训练样本的最大值
        train_array_max = []
        train_array_min = []

        # 分别记录训练样本的数量及特征量数目+1
        n = train_array.shape[0]
        m = train_array.shape[1]

        # 测试样本的数目
        test_n = test_array.shape[0]

        # 提取训练样本和测试样本的特征量和真实结果
        train_x = train_array[:, :m - 1].reshape(n, m - 1)
        train_y = train_array[:, m - 1].reshape(n, )
        test_x = test_array[:, :m - 1].reshape(test_n, m - 1)
        test_y = test_array[:, m - 1].reshape(test_n, )

        # 将特征量归一化
        for i in range(m - 1):
            train_array_max.append(np.max(train_x[:, i]))
            train_array_min.append(np.min(train_x[:, i]))
            if (train_array_max[i] - train_array_min[i]) != 0:
                train_x[:, i] = (train_x[:, i] - train_array_min[i]) / (train_array_max[i] - train_array_min[i])
                test_x[:, i] = (test_x[:, i] - train_array_min[i]) / (train_array_max[i] - train_array_min[i])

        # 利用最邻近算法进行预测训练数据结果
        result = []
        for x1 in test_x:
            distance = []
            for x2 in train_x:
                distance.append(np.sum((x1 - x2) * (x1 - x2)))
            result.append(train_y[distance.index(min(distance))])

        # 获得识别率
        recognition_rate = np.sum((result == test_y)) / len(test_y)
        return recognition_rate, np.array(result).astype('int')

    def SVM(self, train_data, test_data):
        """
        :param train_data:
        :param test_data:
        :return:
        """
        from sklearn.svm import SVC
        svc = SVC(kernel='rbf', gamma='scale')
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        x_train, y_train = train_data[:, :train_data.shape[1] - 1], train_data[:, -1]
        x_test, y_test = test_data[:, :test_data.shape[1] - 1], test_data[:, -1]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        new_train, new_test = Decomposition().PCA(x_train, x_test,
                                                  min(1000, min(x_train.shape[1] - 7, x_train.shape[0])))

        svc.fit(new_train, y_train)
        pred = svc.predict(new_test)

        # 获得识别率
        recognition_rate = np.sum((pred == y_test)) / len(test_data[:, -1])
        print(recognition_rate)

    def MLP(self, train_data, test_data):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        x_train, y_train = train_data[:, :train_data.shape[1] - 1], train_data[:, -1]
        x_test, y_test = test_data[:, :test_data.shape[1] - 1], test_data[:, -1]

        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        hiden_num = int((train_data.shape[1] + 7) * 2 / 3)
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(hiden_num, hiden_num, hiden_num, hiden_num, hiden_num, hiden_num),
                            random_state=1)
        mlp.fit(x_train, y_train)
        pred = mlp.predict(x_test)

        # 获得识别率
        recognition_rate = np.sum((pred == y_test)) / len(test_data[:, -1])
        print(recognition_rate)


class Gabor(object):

    def build_filters(self):
        """
        构建Gabor滤波器
        :return:
        """
        filters = []
        ksize = [3, 5, 7, 9]  # gabor尺度，6个
        lamda = np.pi / 2.0  # 波长
        for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向，0°，45°，90°，135°，共四个
            for K in range(4):
                kern = cv2.getGaborKernel((ksize[K], ksize[K]), 0.56 * ksize[K], theta, lamda, 0.5, 1, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                filters.append(kern)
        return filters

    def process(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def getGabor(self, img, filters, pic_show=False, reduction=1):
        res = []  # 滤波结果
        for i in range(len(filters)):
            res1 = self.process(img, filters[i])
            res1 = Decomposition().mean_pooling(res1, reduction)
            # print(res1.shape)
            res.append(np.asarray(res1))
        if pic_show:
            plt.figure(2)
            for temp in range(len(filters)):
                plt.subplot(4, 4, temp + 1)
                plt.imshow(filters[temp], cmap='gray')
            plt.show()

        return res  # 返回滤波结果,结果为24幅图，按照gabor角度排列


def generate_train(path, result, filters, train):
    files = os.listdir(path)
    for f in files:
        train_img = cv.imread(path + '/' + f, cv.IMREAD_GRAYSCALE)
        res = Gabor().getGabor(train_img, filters)
        res = np.array(res).reshape(-1)
        res = np.append(res, result)
        train.append(res)


def evaluate_valid(data_op=1, op=1, reduction=1, rate=0.2):
    """
    评估函数，在三个数据集上进行评估
    :param op: 1-all 2-part
    :param data_op: 1-CK 2-Fer 3-Jaffe
    :return:
    """
    import preprocess
    from tqdm import tqdm
    from data import CK, Fer2013, Jaffe

    filters = Gabor().build_filters()
    if data_op == 1:
        _, x, y = CK().gen_train_no()
    if data_op == 2:
        _, x, y = Fer2013().gen_train_no()
    if data_op == 3:
        _, x, y = Jaffe().gen_train_no()
    train = []
    if op == 1:
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            x[i] = preprocess.gray_norm(x[i])
            x[i] = preprocess.adaptive_histogram_equalization(x[i])
            res = Gabor().getGabor(x[i], filters, False, reduction)
            res = np.array(res).reshape(-1)
            res = np.append(res, y[i])
            train.append(res)
        train = np.array(train)

    if data_op != 2:
        # 需要划分
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(train, train, random_state=2019, test_size=rate)
        Classifier().SVM(x_train, x_test)
    test1 = []
    test2 = []
    if data_op == 2:
        _, x, y = Fer2013().gen_valid_no(1)
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            x[i] = preprocess.gray_norm(x[i])
            x[i] = preprocess.adaptive_histogram_equalization(x[i])
            res = Gabor().getGabor(x[i], filters, False, reduction)
            res = np.array(res).reshape(-1)
            res = np.append(res, y[i])
            test1.append(res)

        _, x, y = Fer2013().gen_valid_no(2)
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            x[i] = preprocess.gray_norm(x[i])
            x[i] = preprocess.adaptive_histogram_equalization(x[i])
            res = Gabor().getGabor(x[i], filters, False, reduction)
            res = np.array(res).reshape(-1)
            res = np.append(res, y[i])
            test2.append(res)
        test1 = np.array(test1)
        test2 = np.array(test2)
        print("Public")
        Classifier().SVM(train, test1)
        print("Pirvate")
        Classifier().SVM(train, test2)


def evaluate_test():
    """
    在未训练的数据集上进行测试
    :return:
    """
    filters = Gabor().build_filters()
    from tqdm import tqdm
    from data import CK, Fer2013, Jaffe
    _, x, y = Fer2013().gen_train_no()
    train = []
    for i in tqdm(np.arange(0, x.shape[0], 1)):
        x[i] = preprocess.gray_norm(x[i])
        x[i] = preprocess.adaptive_histogram_equalization(x[i])
        res = Gabor().getGabor(x[i], filters, False, 6)
        res = np.array(res).reshape(-1)
        res = np.append(res, y[i])
        train.append(res)
    train = np.array(train)

    test = []
    _, x, y = Jaffe().gen_train_no()
    for i in tqdm(np.arange(0, x.shape[0], 1)):
        x[i] = preprocess.gray_norm(x[i])
        x[i] = preprocess.adaptive_histogram_equalization(x[i])
        res = Gabor().getGabor(x[i], filters, False, 6)
        res = np.array(res).reshape(-1)
        res = np.append(res, y[i])
        test.append(res)
    test = np.array(train)

    Classifier().SVM(train, test)

    test = []
    _, x, y = CK().gen_train_no()
    for i in tqdm(np.arange(0, x.shape[0], 1)):
        x[i] = preprocess.gray_norm(x[i])
        x[i] = preprocess.adaptive_histogram_equalization(x[i])
        res = Gabor().getGabor(x[i], filters, False, 6)
        res = np.array(res).reshape(-1)
        res = np.append(res, y[i])
        test.append(res)
    test = np.array(train)
    Classifier().SVM(train, test)


if __name__ == '__main__':
    # 在本数据集上训练并评估
    # 0.9645 0.949 (784, 36865) (197, 36865) re = 6
    print("CK+:")
    evaluate_valid(1, 1, 3)
    # public: 0.458 private : 0.389
    # print("Fer20 13:")
    # evaluate_valid(2, 1, 8)
    # # 0.4186 0.697
    # print("Jaffe")
    # evaluate_valid(3, 2, 1, 0.1)
    # # 在不同数据集上训练并评估
    # # Jaffe 0.705 CK: 0.705
    # evaluate_test()
