from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import tensorflow as tf
import keras
from tensorflow import nn

from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy

from keras import layers, Sequential
from keras.models import Model as ModelClass
from keras.callbacks import EarlyStopping, Callback
from keras.preprocessing.image import ImageDataGenerator

curr_path = os.getcwd()

class AccurayCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

@keras.saving.register_keras_serializable('my_package')
class FFTLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Apply FFT
        fft_result = tf.signal.fft2d(tf.cast(inputs, dtype=tf.complex64))

        # Shift FFT
        fft_shifted = tf.signal.fftshift(fft_result)

        # Compute log magnitude
        log_magnitude = tf.math.log(tf.abs(fft_shifted) + 10e-8)

        # Flatten the log magnitude
        flattened_log_magnitude = layers.Flatten()(log_magnitude)

        # Expand dimension
        expanded_log_magnitude = tf.expand_dims(flattened_log_magnitude, axis=-1)

        return expanded_log_magnitude

    def get_config(self):
        config = super(FFTLayer, self).get_config()
        return config

class Model():
    def __init__(self) -> None:
        self.model = Sequential([
            layers.Input(shape=(227, 227, 3)),
            FFTLayer(),

            layers.Conv1D(filters=32, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.Activation(activation=nn.relu),
            layers.MaxPooling1D(pool_size=(4,), strides=2),

            layers.Conv1D(filters=64, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.Activation(activation=nn.relu),
            layers.MaxPooling1D(pool_size=(4,), strides=4),

            layers.Conv1D(filters=128, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.Activation(activation=nn.relu),
            layers.MaxPooling1D(pool_size=(4,), strides=4),

            layers.Conv1D(filters=128, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.Activation(activation=nn.relu),
            layers.MaxPooling1D(pool_size=(4,), strides=4),

            layers.Bidirectional(layers.LSTM(units=128)),
            layers.Dense(units=128, activation=nn.sigmoid),
            layers.Dense(units=1, activation=nn.sigmoid)
        ])
        self.model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(name='accuracy')])

        self.acc_callback = AccurayCallback()
        self.early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
        self.train_history = None

        self.train_data_generator = None
        self.val_data_generator = None
    
    def train(self, num_epochs=1000) -> None:
        train_datagen = ImageDataGenerator(
            1./255,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(
            rescale=1./255
        )

        self.train_data_generator = train_datagen.flow_from_directory(
            directory=os.path.join(curr_path, 'data', 'train'),
            target_size=(227, 227),
            color_mode='rgb',
            class_mode='binary',
            batch_size=4
        )
        self.val_data_generator = val_datagen.flow_from_directory(
            directory=os.path.join(curr_path, 'data', 'validation'),
            target_size=(227, 227),
            color_mode='rgb',
            class_mode='binary',
            batch_size=1
        )

        self.train_history = self.model.fit(self.train_data_generator, validation_data=self.val_data_generator, callbacks=[self.acc_callback, self.early_stop_callback], epochs=num_epochs)
        self.model.save(filepath=os.path.join(curr_path, 'models', 'conv1D_lstm_model.keras'))
        print(f"Model saved")
    
    def plot_train(self) -> None:

        plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)

        plt.subplot(1, 2, 1)
        plt.plot(self.train_history.history['loss'])
        plt.title('Train Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')

        plt.subplot(1, 2, 2)
        plt.plot(self.train_history.history['accuracy'])
        plt.title('Train Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('epoch')
        plt.show()

if __name__ == '__main__':
    model = Model()
    model.train(num_epochs=1)
    model.plot_train()