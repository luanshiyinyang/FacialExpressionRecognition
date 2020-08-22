"""
author: Zhou Chen
datetime: 2019/7/1 20:24
desc: 使用LBP的实现
"""


import preprocess
import sys
from Gabor import *
import os
import numpy as np
import cv2
from skimage import feature as skif
from sklearn import svm


class LBP(object):
    # 使用LBP+SVM实现表情识别

    def load_image(self, foler):
        """
        根据给定的目录，读取当前目录下的所有图片
        :param foler:
        :return:
        """
        categories = os.listdir(foler)  # 得到当前foler文件夹下所有的目录
        imags = []
        labels = []
        for category in categories:
            now_folder = os.path.join(foler, category)
            subsdirectories = os.listdir(now_folder)
            for subsdirectory in subsdirectories:
                now_path = os.path.join(now_folder, subsdirectory)
                image = cv2.imread(now_path)
                grab = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                imags.append(self.get_lbp(grab))
                labels.append(category)
        images_array = np.array(imags)
        return images_array, labels

    def get_lbp(self, image):
        """
        获取给定图片的LBP，划分成几个区域后
        :param rgb:
        :return:
        """
        gridx = 6
        gridy = 6
        widx = 8
        widy = 8
        hists = []
        for i in range(gridx):
            for j in range(gridy):
                mat = image[i * widx: (i + 1) * widx, j * widy: (j + 1) * widy]
                lbp = skif.local_binary_pattern(mat, 8, 1, 'uniform')
                max_bins = 10
                hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
                hists.append(hist)
        out = np.array(hists).reshape(-1, 1)
        return out

    def label2number(self, label_list):
        label = np.zeros(len(label_list))
        label_unique = np.unique(label_list)
        num = label_unique.shape[0]
        for k in range(num):
            index = [i for i, v in enumerate(label_list) if v == label_unique[k]]
            label[index] = k
        return label, label_unique

    def train_test(self, train_data, train_label, test_data, test_label):
        train_data = np.squeeze(train_data, axis=-1)
        test_data = np.squeeze(test_data, axis=-1)
        svr_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='auto')
        svr_rbf.fit(train_data, train_label)
        pred = svr_rbf.predict(test_data)
        print(np.sum(pred == test_label))


def evaluate_lbp(data_op=1, op=1, reduction=1, rate=0.2):
    """
    评估函数，在三个数据集上进行评估
    :param op: 1-all 2-part
    :param data_op: 1-CK 2-Fer 3-Jaffe
    :return:
    """
    filters = Gabor().build_filters()
    from tqdm import tqdm
    from data import CK, Fer2013, Jaffe
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
    if op == 2:
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            img, dets, shape_list, img_list, pt_post_list = preprocess.deal(x[i])
            res1 = Gabor().getGabor(img, filters, 0, 1)
            res1 = np.array(res1)
            res = []
            if len(shape_list) == 0:
                continue
            for _, pt in enumerate(shape_list[0].parts()):
                px, py = min(max(pt.x, 0), 47), min(max(pt.y, 0), 47)
                im = res1[0]
                cv2.circle(im, (px, py), 2, (255, 0, 0), 1)
                res.append(res1[:, px, py])
            res = np.array(res)
            res = np.append(res, y[i])
            train.append(res)
        train = np.array(train)
    if op == 3:
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            res = LBP().get_lbp(x[i])
            res = np.array(res).reshape(-1)
            res = np.append(res, y[i])
            train.append(res)
        train = np.array(train)
    if data_op != 2:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(train, train, random_state=2019, test_size=rate)
        Classifier().SVM(x_train, x_test)
    test1 = []
    test2 = []
    if data_op == 2:
        _, x, y = Fer2013().gen_valid_no(1)
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            res = LBP().get_lbp(x[i])
            res = np.array(res).reshape(-1)
            res = np.append(res, y[i])
            test1.append(res)

        _, x, y = Fer2013().gen_valid_no(2)
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            res = LBP().get_lbp(x[i])
            res = np.array(res).reshape(-1)
            res = np.append(res, y[i])
            test2.append(res)
        test1 = np.array(test1)
        test2 = np.array(test2)
        print("Public")
        Classifier().SVM(train, test1)
        print("Pirvate")
        Classifier().SVM(train, test2)


# 在未训练的数据集上进行测试
def evaluate1_lbp():
    filters = Gabor().build_filters()
    from tqdm import tqdm
    from data import CK, Fer2013, Jaffe
    _, x, y = Fer2013().gen_train_no()
    train = []
    for i in tqdm(np.arange(0, x.shape[0], 1)):
        res = LBP().get_lbp(x[i])
        res = np.array(res).reshape(-1)
        res = np.append(res, y[i])
        train.append(res)
    train = np.array(train)

    test = []
    _, x, y = Jaffe().gen_train_no()
    for i in tqdm(np.arange(0, x.shape[0], 1)):
        res = LBP().get_lbp(x[i])
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


if __name__ == "__main__":
    # 在本数据集上训练并评估
    # 0.9645 0.949 (784, 36865) (197, 36865) re = 6
    print("CK+:")
    evaluate_lbp(1, 3, 3)
    # public: 0.458 private : 0.389
    # print("Fer2013:")
    # evaluate_lbp(2, 3, 8)
    # 0.4186 0.697
    # print("Jaffe")
    # evaluate_lbp(3, 3, 1, 0.1)
    # 在不同数据集上训练并评估
    # Jaffe 0.705 CK: 0.705
    # evaluate1_lbp()
