# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# import tensorflow as tf
import pickle

import cv2
import numpy as np
import random

from PIL import Image
from keras import regularizers
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
import tflearn.datasets.oxflower17 as oxflower17
import keras
from keras.models import Sequential
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from tensorflow.examples.tutorials.mnist import input_data
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
# session = tf.Session(config=config)
#
# # 设置session
# KTF.set_session(session)


# 函数调用：生成数据集
def initPKL(imgSet_shuffle, train_or_test):
    imgSet = []
    labels = []
    label_names = []

    if train_or_test == 'train':
        set_name = 'trainSet.pkl'
    else:
        set_name = 'testSet.pkl'

    for i in imgSet_shuffle:
        imgSet.append(i[0])
        labels.append(i[1])
        label_names.append(i[2])

    imgSet = np.array(imgSet)
    labels = np.array(labels)
    label_names = np.array(label_names)
    arr = (imgSet, labels, label_names)

    # 写入文件
    data = (arr[0], arr[1], arr[2])
    output = open(set_name, 'wb')
    pickle.dump(data, output)
    output.close()


def initArr(folders_path):
    i = 0
    imgSet = []
    folders = os.listdir(folders_path)

    for folder in folders:
        # 类别个数,几个0代表几类
        label = [0, 0]
        files = os.listdir(folders_path + folder)
        label[i] = 1
        for file in files:
            # 读取图片
            img_arr = np.array(Image.open(folders_path + folder + '/' + file)) / 255

            imgSet.append((img_arr, label, folder))
        i += 1
    return imgSet


def test():
    # 将图片转换成数组
    train_folders_path = r'E:\PythonProject\Machine learning\CNN\17flowers\jpg'
    # test_folders_path = 'E:/workFolder/data/cifar/cifar_10/test/'

    train_imgSet = initArr(train_folders_path)
    # test_imgSet = initArr(test_folders_path)

    # 打乱顺序
    random.shuffle(train_imgSet)
    # random.shuffle(test_imgSet)

    train_set_shuffle = np.array(train_imgSet)
    # test_set_shuffle = np.array(test_imgSet)

    # 分别生成训练集和测试集
    initPKL(train_set_shuffle, 'train')
    # initPKL(test_set_shuffle, 'test')

    # 测试生成的数据集
    f = open('./trainSet.pkl', 'rb')
    x, y, z = pickle.load(f)
    f.close()


# define model
def getVGG16Model(input_shape, num_classes, gpuNum=2):
    weight_decay = 0.0005
    model = Sequential()
    # =======================第一层===============================
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 224
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 224
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

    # =======================第二层===============================
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 112
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 112
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

    # =======================第三层===============================
    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 110

    # =======================第四层===============================
    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(weight_decay)))  # 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # =======================第五层===============================
    # 减少一层
    # model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
    #                  kernel_regularizer=regularizers.l2(weight_decay)))  # 14
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
    #                  kernel_regularizer=regularizers.l2(weight_decay)))  # 14
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, kernel_size=(3, 3), padding='same',
    #                  kernel_regularizer=regularizers.l2(weight_decay)))  # 14
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # 7

    # =======================全连接层===============================
    # model.add(GlobalAveragePooling2D())  # straightening the output
    # model.add(Flatten())  # straightening the output
    # model.add(Dense(4096, activation='relu'))  # the link layera
    # model.add(Dropout(0.3))
    # model.add(Dense(4096, activation='relu'))  # the link layer
    # model.add(Dense(num_classes, activation='softmax'))

    # 改变全连接层
    model.add(Flatten())  # straightening the output
    model.add(Dense(512, activation='relu', name='fc'))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))

    # parallel_model = multi_gpu_model(model, gpus=gpuNum)
    # define loss function,optimization
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(epsilon=1e-08),
                  metrics=['accuracy'])
    return model


def getMnist():
    # 读取数据集
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'  # 训练集图像的文件名
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'  # 训练集label的文件名
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'  # 测试集图像的文件名
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'  # 测试集label的文件名
    local_file = r'E:\PythonProject\Machine learning\CNN\mnist'
    # (x_train, y_train), (x_test, y_test) = mnist.load_data("../test_data_home")

    mnist = input_data.read_data_sets(local_file, one_hot=True)
    trainX = mnist.train.images
    trainY = mnist.train.labels
    testX = mnist.test.images
    testY = mnist.test.labels

    # trainX, trainY = trainX[:1000], trainY[:1000]
    # testX, testY = testX[:1000], testY[:1000]
    # GRAY两通道转换为RGB三通道
    trainX = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in trainX]
    testX = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in testX]

    trainX = np.concatenate([arr[np.newaxis] for arr in trainX]).astype('float32')
    testX = np.concatenate([arr[np.newaxis] for arr in testX]).astype('float32')

    trainX /= 255
    testX /= 255
    return trainX, trainY, testX, testY


def getFlower():
    # X, Y = oxflower17.load_data(one_hot=True)  # 下载oxflower17数据集
    folders_path = r'E:\PythonProject\Machine learning\CNN\17flowers\jpg' + '\\'
    imgSet = []
    files = os.listdir(folders_path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []

    for file in files:
        # 读取图片
        img_arr = np.array(Image.open(folders_path + file).resize((224, 224)))
        per_image_Rmean.append(np.mean(img_arr[:, :, 0]))
        per_image_Gmean.append(np.mean(img_arr[:, :, 1]))
        per_image_Bmean.append(np.mean(img_arr[:, :, 2]))
        imgSet.append(img_arr)

    R_mean = np.mean(per_image_Rmean)
    print("r mean")
    print(R_mean.shape)

    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)

    X = np.array(imgSet)
    # print(X.shape)
    X = X - np.mean(X, axis=0)
    # print(np.mean(X, axis=0))
    Y = np.array([i // 80 for i in range(X.shape[0])])

    # turn to one-hot code
    Y = keras.utils.to_categorical(Y, len(np.unique(Y)))
    # testY = keras.utils.to_categorical(testY,num_classes)
    trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=0)
    validX, testX, validY, testY = train_test_split(testX, testY, train_size=0.3, test_size=0.7, random_state=0)
    return trainX, trainY, validX, validY, testX, testY


if __name__ == '__main__':
    getFlower()
    # a1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(a1.shape)
