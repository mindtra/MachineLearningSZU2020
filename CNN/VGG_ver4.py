import argparse
from keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dense, Flatten
)

import os
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import plot_model
from keras.models import load_model


# utils
def normalize_image_array(image_array):
    # N, D = image_array.shape

    numerator = image_array - np.expand_dims(np.mean(image_array, 1), 1)
    denominator = np.expand_dims(np.std(image_array, 1), 1)

    return numerator / (denominator + 1e-7)


def load_mnist(samplewise_normalize=True):
    mnist = input_data.read_data_sets(r'E:\PythonProject\Machine learning\CNN\mnist', one_hot=True)

    train_X = mnist.train.images
    train_y = mnist.train.labels

    valid_X = mnist.validation.images
    valid_y = mnist.validation.labels

    test_X = mnist.test.images
    test_y = mnist.test.labels

    if samplewise_normalize:
        train_X = normalize_image_array(train_X)
        valid_X = normalize_image_array(valid_X)
        test_X = normalize_image_array(test_X)

    train_X = np.reshape(train_X, [-1, 28, 28, 1])
    valid_X = np.reshape(valid_X, [-1, 28, 28, 1])
    test_X = np.reshape(test_X, [-1, 28, 28, 1])

    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


def train_generator():
    train_gen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    val_gen = ImageDataGenerator()
    return train_gen, val_gen


def plot_(model_path, file_path):
    model = load_model(model_path)
    plot_model(model,
               file_path,
               show_shapes=True,
               show_layer_names=False)


# vgg16
def vgg(input_tensor):
    def two_conv_pool(x, F1, F2, name):
        x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

        return x

    def three_conv_pool(x, F1, F2, F3, name):
        x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(F3, (3, 3), activation=None, padding='same', name='{}_conv3'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

        return x

    net = input_tensor

    net = two_conv_pool(net, 64, 64, "block1")
    net = two_conv_pool(net, 128, 128, "block2")
    net = three_conv_pool(net, 256, 256, 256, "block3")
    net = three_conv_pool(net, 512, 512, 512, "block4")

    net = Flatten()(net)
    net = Dense(512, activation='relu', name='fc')(net)
    net = Dense(10, activation='softmax', name='predictions')(net)

    return net


# basemodel
class BaseModel(object):

    def __init__(self, name, fn, model_path):
        """Constructor for BaseModel
        Parameters
        ----------
        name : str
            Name of this model
        fn : function
            Inference function, y = fn(X)
        model_path : str
            Path to a model.h5
        """
        X = Input(shape=[28, 28, 1])
        y = fn(X)

        self.model = Model(X, y, name=name)
        self.model.compile("adam", "categorical_crossentropy", ["accuracy"])

        self.path = model_path
        self.name = name
        self.load()

    def fit(self, train_data, valid_data, epochs=10, batchsize=32, **kwargs):
        """Training function
        Evaluate at each epoch against validation data
        Save the best model according to the validation loss
        Parameters
        ----------
        train_data : tuple, (X_train, y_train)
            X_train.shape == (N, H, W, C)
            y_train.shape == (N, N_classes)
        valid_data : tuple
            (X_val, y_val)
        epochs : int
            Number of epochs to train
        batchsize : int
            Minibatch size
        **kwargs
            Keywords arguments for `fit_generator`
        """
        callback_best_only = ModelCheckpoint(self.path, save_best_only=True)
        train_gen, val_gen = train_generator()

        X_train, y_train = train_data
        X_val, y_val = valid_data

        N = X_train.shape[0]
        N_val = X_val.shape[0]

        self.model.fit_generator(train_gen.flow(X_train, y_train, batchsize),
                                 steps_per_epoch=N / batchsize,
                                 validation_data=val_gen.flow(X_val, y_val, batchsize),
                                 validation_steps=N_val / batchsize,
                                 epochs=epochs,
                                 callbacks=[callback_best_only],
                                 **kwargs)

    def save(self):
        """Save weights
        Should not be used manually
        """
        self.model.save_weights(self.path)

    def load(self):
        """Load weights from self.path """
        if os.path.isfile(self.path):
            self.model.load_weights(self.path)
            print("Model loaded")
        else:
            print("No model is found")

    def predict(self, X):

        return self.model.predict(X)

    def evaluate(self, X, y):

        return self.model.evaluate(X, y)


class VGGNet(BaseModel):
    def __init__(self, model_path):
        super(VGGNet, self).__init__("VGG", vgg, model_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", type=int, default=32, help="Epochs")
    parser.add_argument("--model_path", default="model", type=str, help="model path ")

    args = parser.parse_args()
    return args.epoch, args.model_path


def main():
    # EPOCH, MODEL_PATH = arg_parser()
    for i in range(10000):
        # (X, y)
        # train, valid, _ = load_mnist(samplewise_normalize=True)
        train, valid, test = load_mnist(samplewise_normalize=True)
        vggnet = VGGNet("model")
        vggnet.model.summary()
        # vggnet.fit((train[0], train[1]), (valid[0], valid[1]), 32)
        vggnet.fit((train[0], train[1]), (valid[0], valid[1]), 1, 128)
        score = vggnet.evaluate(test[0], test[1])
        print('test loss:', score[0])
        print('test accuracy:', score[1] * 100)

        i = i + 1


if __name__ == '__main__':
    main()
