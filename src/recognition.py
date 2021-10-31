"""
author: Zhou Chen
datetime: 2019/6/19 18:49
desc: 本模块为表情预测处理模块
"""
import os
import cv2
import numpy as np
from utils import index2emotion, expression_analysis, cv2_img_add_text
from blazeface import blaze_detect



def face_detect(img_path, model_selection="default"):
    """
    检测测试图片的人脸
    :param img_path: 图片的完整路径
    :return:
    """
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if model_selection == "default":
        face_cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30)
        )
    elif model_selection == "blazeface":
        faces = blaze_detect(img)
    else:
        raise NotImplementedError("this face detector is not supported now!!!")

    return img, img_gray, faces


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """
    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    # resized_images.append(cv2.flip(face_img[2], 1))
    # resized_images.append(cv2.flip(face_img[3], 1))
    # resized_images.append(cv2.flip(face_img[4], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression(img_path, model):
    """
    对图中n个人脸进行表情预测
    :param img_path:
    :return:
    """

    border_color = (0, 0, 0)  # 黑框框
    font_color = (255, 255, 255)  # 白字字

    img, img_gray, faces = face_detect(img_path, 'blazeface')
    if len(faces) == 0:
        return 'no', [0, 0, 0, 0, 0, 0, 0, 0]
    # 遍历每一个脸
    emotions = []
    result_possibilitys = []
    for (x, y, w, h) in faces:
        face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
        faces_img_gray = generate_faces(face_img_gray)
        # 预测结果线性加权
        results = model.predict(faces_img_gray)
        result_sum = np.sum(results, axis=0).reshape(-1)
        label_index = np.argmax(result_sum, axis=0)
        emotion = index2emotion(label_index, 'en')
        
        cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
        img = cv2_img_add_text(img, emotion, x + 30, y + 30, font_color, 20)
        emotions.append(emotion)
        result_possibilitys.append(result_sum)
    if not os.path.exists("./output"):
        os.makedirs("./output")
    cv2.imwrite('./output/rst.png', img)
    return emotions[0], result_possibilitys[0]


if __name__ == '__main__':
    from model import CNN3
    model = CNN3()
    model.load_weights('./models/cnn3_best_weights.h5')
    predict_expression('./input/test/happy2.png', model)
