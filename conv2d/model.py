from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors

from keras import layers
from keras.models import Model
from tensorflow import nn
from keras.callbacks import Callback

from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

class AccurayCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True

class CrackDetector(Model):

    def __init__(self):
        super().__init__()

        self.conv2d_1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, input_shape=(227, 227, 3))
        self.BN_1 = layers.BatchNormalization()
        self.activation_1 = layers.Activation(activation=nn.relu)
        self.Maxpooling_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=1)

        self.conv2d_2 = layers.Conv2D(filters=64, kernel_size=3, strides=1)
        self.BN_2 = layers.BatchNormalization()
        self.activation_2 = layers.Activation(activation=nn.relu)
        self.Maxpooling_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv2d_3 = layers.Conv2D(filters=128, kernel_size=3, strides=1)
        self.BN_3 = layers.BatchNormalization()
        self.activation_3 = layers.Activation(activation=nn.relu)
        self.Maxpooling_3 = layers.MaxPooling2D(pool_size=(4, 4), strides=4)

        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(units=128, activation=nn.sigmoid)
        self.dense_2 = layers.Dense(units=1, activation=nn.sigmoid)

    def call(self, inputs, training=False):
        
        x = self.conv2d_1 (inputs)
        x = self.BN_1 (x)
        x = self.activation_1 (x)
        x = self.Maxpooling_1 (x)

        x = self.conv2d_2 (x)
        x = self.BN_2 (x)
        x = self.activation_2 (x)
        x = self.Maxpooling_2 (x)

        x = self.conv2d_3 (x)
        x = self.BN_3 (x)
        x = self.activation_3 (x)
        x = self.Maxpooling_3 (x)

        x = self.flatten (x)
        if training:
            x = layers.Dropout(rate=0.5)(x)

        x = self.dense_1(x)
        if training:
            x = layers.Dropout(rate=0.5)(x)

        outputs = self.dense_2 (x)

        return outputs

def build_model():
    model = CrackDetector()
    model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[BinaryAccuracy(name='accuracy')])

    return model

if __name__ == '__main__':
    pass