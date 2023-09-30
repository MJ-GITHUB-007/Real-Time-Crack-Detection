from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors

import os
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from model import build_model, AccurayCallback

curr_path = os.getcwd()

train_dategen = ImageDataGenerator(
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

train_data_generator = train_dategen.flow_from_directory(
    directory=os.path.join(curr_path, 'data', 'train'),
    target_size=(227, 227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=4
)
val_data_generator = train_dategen.flow_from_directory(
    directory=os.path.join(curr_path, 'data', 'validation'),
    target_size=(227, 227),
    color_mode='rgb',
    class_mode='binary',
    batch_size=1
)

acc_callback = AccurayCallback()
early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
num_epochs = 1000

model = build_model()
train_history = model.fit(train_data_generator, validation_data=val_data_generator, epochs=num_epochs, callbacks=[acc_callback, early_stop_callback])
model.save(filepath=os.path.join(curr_path, 'models', 'conv2D_model.keras'))
print(f"Model saved")

plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)

plt.subplot(1, 2, 1)
plt.plot(train_history.history['loss'])
plt.title('Train Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(train_history.history['accuracy'])
plt.title('Train Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.show()