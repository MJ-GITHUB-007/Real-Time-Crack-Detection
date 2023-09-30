from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors

from keras import layers, Sequential
from tensorflow import nn
from keras.callbacks import Callback

from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import os

curr_path = os.getcwd()

class AccurayCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

class Model():
    def __init__(self) -> None:
        self.model = Sequential([
            layers.Input(shape=(227, 227, 3)),

            layers.Conv2D(filters=32, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.Activation(activation=nn.relu),
            layers.MaxPooling2D(pool_size=(2, 2), strides=1),

            layers.Conv2D(filters=64, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.Activation(activation=nn.relu),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),

            layers.Conv2D(filters=128, kernel_size=3, strides=1),
            layers.BatchNormalization(),
            layers.Activation(activation=nn.relu),
            layers.MaxPooling2D(pool_size=(4, 4), strides=4),

            layers.Flatten(),
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
        self.model.save(filepath=os.path.join(curr_path, 'models', 'conv2D_model.keras'))
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