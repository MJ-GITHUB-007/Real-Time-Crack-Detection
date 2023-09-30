import numpy as np
import cv2
import os

import tensorflow as tf
from tensorflow import nn
from keras import layers
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy
from keras.models import Model
from keras.callbacks import EarlyStopping, Callback

curr_path = os.getcwd()

class AccurayCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

class CrackDetector(Model):

    def __init__(self):
        super().__init__()

        self.conv1d_1 = layers.Conv1D(filters=32, kernel_size=3, strides=1)
        self.BN_1 = layers.BatchNormalization()
        self.activation_1 = layers.Activation(activation=nn.relu)
        self.Maxpooling_1 = layers.MaxPooling1D(pool_size=(4,), strides=2)

        self.conv1d_2 = layers.Conv1D(filters=64, kernel_size=3, strides=1)
        self.BN_2 = layers.BatchNormalization()
        self.activation_2 = layers.Activation(activation=nn.relu)
        self.Maxpooling_2 = layers.MaxPooling1D(pool_size=(4,), strides=4)

        self.conv1d_3 = layers.Conv1D(filters=128, kernel_size=3, strides=1)
        self.BN_3 = layers.BatchNormalization()
        self.activation_3 = layers.Activation(activation=nn.relu)
        self.Maxpooling_3 = layers.MaxPooling1D(pool_size=(4,), strides=4)

        self.conv1d_4 = layers.Conv1D(filters=128, kernel_size=3, strides=1)
        self.BN_4 = layers.BatchNormalization()
        self.activation_4 = layers.Activation(activation=nn.relu)
        self.Maxpooling_4 = layers.MaxPooling1D(pool_size=(4,), strides=4)

        self.lstm_1 = layers.Bidirectional(layers.LSTM(units=128))
        self.dense_1 = layers.Dense(units=128, activation=nn.sigmoid)
        self.dense_2 = layers.Dense(units=1, activation=nn.sigmoid)

    def call(self, inputs, training=False):
        # print(type(inputs))
        # inputs = inputs.numpy()
        # print(type(inputs))
        # inputs = tf.convert_to_tensor(inputs)
        # print(type(inputs))
        # exit()
        x = self.__get_ffts(inputs)

        x = self.conv1d_1 (x)
        x = self.BN_1 (x)
        x = self.activation_1 (x)
        x = self.Maxpooling_1 (x)

        x = self.conv1d_2 (x)
        x = self.BN_2 (x)
        x = self.activation_2 (x)
        x = self.Maxpooling_2 (x)

        x = self.conv1d_3 (x)
        x = self.BN_3 (x)
        x = self.activation_3 (x)
        x = self.Maxpooling_3 (x)

        x = self.conv1d_4 (x)
        x = self.BN_4 (x)
        x = self.activation_4 (x)
        x = self.Maxpooling_4 (x)

        x = self.lstm_1 (x)
        if training:
            x = layers.Dropout(rate=0.5)(x)

        x = self.dense_1 (x)
        if training:
            x = layers.Dropout(rate=0.5)(x)

        outputs = self.dense_2 (x)
        return outputs
    
    def __get_ffts(self, matrices):
        matrices = matrices.numpy()
        new_matrices = []
        for matrix in matrices:
            matrix = cv2.cvtColor(matrix, cv2.COLOR_RGB2GRAY)
            matrix = np.fft.fft2(matrix)
            matrix = np.fft.fftshift(matrix)
            matrix = np.log(np.abs(matrix) + 1)
            matrix = matrix.flatten()
            matrix = np.expand_dims(matrix, axis=1)
            new_matrices.append(matrix)
        new_matrices = tf.convert_to_tensor(new_matrices)
        return new_matrices


def build_model():
    model = CrackDetector()
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(name='accuracy')], run_eagerly=True)

    return model
