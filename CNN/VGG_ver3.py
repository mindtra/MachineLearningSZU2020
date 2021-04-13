# -*- coding: utf-8 -*-
import os
import keras
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD
import gzip
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

# from tflearn.datasets.mnist import extract_images, extract_labels
from CNN.VGG_ver2 import getFlower, getVGG16Model
from dataprocess import normalize_image_array, get_data, separateData

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

imgSize = 46
n_class = 40
batch_size = 32
inputSize = (imgSize, imgSize, 1)


def getVGG16ver3():
    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=inputSize)

    for layer in model_vgg.layers:
        layer.trainable = False

    model = Flatten(name='flatten')(model_vgg.output)  # 扁平化
    # model = BatchNormalization(epsilon=1e-6)(model)
    model = Dense(4096, activation='relu', name='fc1')(model)
    # model = BatchNormalization(epsilon=1e-6)(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(0.5)(model)
    model = Dense(n_class, activation='softmax')(model)
    model_vgg_mnist = Model(inputs=model_vgg.input, outputs=model, name='vgg16')

    model_vgg_mnist.summary()

    # VGGNet初始推荐
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9)  # 随机梯度下降
    model_vgg_mnist.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model_vgg_mnist


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'g', label='train accuracy')
        # loss
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val accuracy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def getORL():
    data, n_class, n_one_class, _ = get_data()
    train_data, train_data_index, test_data, test_data_index = separateData(data, 6, 0, n_class)
    # validX, testX, validY, testY = train_test_split(test_data, test_data_index, train_size=0.4, test_size=0.6,
    #                                                 random_state=0)
    train_data_index = keras.utils.to_categorical(train_data_index, n_class)
    test_data_index = keras.utils.to_categorical(test_data_index, n_class)

    size = 46
    validNum = 1
    testNum = n_one_class - 6 - validNum
    validX = np.zeros((n_class * validNum, size * size))
    validY = np.zeros((n_class * validNum, n_class))
    testX = np.zeros((n_class * testNum, size * size))
    testY = np.zeros((n_class * testNum, n_class))
    valid_num = 0  # 训练集的个数
    test_num = 0  # 测试集的个数
    for j in range(test_data.shape[0]):
        if j % (n_one_class - 6) < validNum:
            validX[valid_num] = test_data[j]
            validY[valid_num] = test_data_index[j]
            valid_num += 1
        else:
            testX[test_num] = test_data[j]
            testY[test_num] = test_data_index[j]
            test_num += 1

    print(train_data.shape)
    print(validX.shape)
    print(testX.shape)

    train_data = normalize_image_array(train_data)
    validX = normalize_image_array(validX)
    testX = normalize_image_array(testX)

    train_data = np.reshape(train_data, [-1, size, size, 1])
    validX = np.reshape(validX, [-1, size, size, 1])
    testX = np.reshape(testX, [-1, size, size, 1])
    print(train_data.shape)
    print(validX.shape)
    print(testX.shape)
    print(train_data_index.shape)
    print(validY.shape)
    print(testY.shape)
    return train_data, train_data_index, validX, validY, testX, testY


def getMnistV3(localFile):
    mnist = input_data.read_data_sets(localFile, one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    x_train = normalize_image_array(x_train)
    x_valid = normalize_image_array(x_valid)
    x_test = normalize_image_array(x_test)

    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)

    # # GRAY两通道转换为RGB三通道
    # x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
    # x_valid = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_valid]
    # x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
    #
    # x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')
    # x_valid = np.concatenate([arr[np.newaxis] for arr in x_valid]).astype('float32')
    # x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

    # x_train = x_train / 255
    # x_test = x_test / 255

    x_train = np.reshape(x_train, [-1, 28, 28, 1])
    x_valid = np.reshape(x_valid, [-1, 28, 28, 1])
    x_test = np.reshape(x_test, [-1, 28, 28, 1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test


if __name__ == '__main__':
    # 读取数据集
    # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'  # 训练集图像的文件名
    # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'  # 训练集label的文件名
    # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'  # 测试集图像的文件名
    # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'  # 测试集label的文件名
    for i in range(20):
        local_file = r'E:\PythonProject\Machine learning\CNN\mnist'
        # (x_train, y_train), (x_test, y_test) = mnist.load_data("../test_data_home")
        x_train, y_train, x_valid, y_valid, x_test, y_test = getORL()
        # x_train, y_train, x_valid, y_valid, x_test, y_test = getMnistV3(local_file)

        print(x_train.shape)
        print(x_valid.shape)
        print(x_test.shape)
        train_datagen = ImageDataGenerator(rotation_range=30, zoom_range=0.2,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1, horizontal_flip=True)
        val_datagen = ImageDataGenerator()

        train_datagen.fit(x_train)
        val_datagen.fit(x_valid)

        train_datagenerator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
        validation_generator = val_datagen.flow(x_valid, y_valid, batch_size=batch_size)

        # 加载已有的模型
        path = 'model_weight_ORL_BN_adamv4.h5'
        # path = 'model_weight_minist_BN_adamv4.h5'
        # model_vgg_mnist = load_model('model_weight_minist.h5')
        # model_vgg_mnist = load_model('model_weight_minist_BN.h5')  # 1加入BN
        # model_vgg_mnist = load_model('model_weight_minist_BN_adam.h5')  # 2用adam
        # model_vgg_mnist = load_model('model_weight_minist_BN_adamv2.h5')  # 1+2+3更换全连接
        # model_vgg_mnist = load_model('model_weight_minist_BN_adamv3.h5')  # 1+2+3+不改变原图,减少层数
        # model_vgg_mnist = load_model('model_weight_minist_BN_adamv4.h5')  # 1+2+不改变原图,减少层数,不改变全连接
        model_vgg_mnist = getVGG16Model(inputSize, n_class)
        if os.path.isfile(path):
            model_vgg_mnist = load_model(path)
            print("Model loaded")
        else:
            print("No model")

        # callback 函数
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5,
                                                    min_lr=0.0001)
        callback_best_only = ModelCheckpoint(path, save_best_only=True)
        # history = LossHistory()
        model_vgg_mnist.summary()
        model_vgg_mnist.fit_generator(generator=train_datagenerator, steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=10, validation_data=validation_generator,
                                      callbacks=[callback_best_only, learning_rate_reduction])

        # x_train, y_train, x_test, y_test = getFlower()
        # model_vgg_mnist.fit(x_train, y_train, validation_split=0.1, epochs=1000, batch_size=100)
        # model_vgg_mnist.save('model_weight_minist_BN_adamv3.h5')

        # 以下为临时文件，无用删除
        model_vgg_mnist.save('model_weight_ORL_BN_adamv4.h5')
        # model_vgg_mnist.save('model_weight_minist_BN_adamv4.h5')

        score = model_vgg_mnist.evaluate(x_test, y_test, verbose=1)
        print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (score[0], score[1] * 100))
        score = model_vgg_mnist.evaluate(x_train, y_train, verbose=1)
        print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (score[0], score[1] * 100))
        # history.loss_plot('batch')
