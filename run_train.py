from warnings import filterwarnings
filterwarnings(action='ignore')
import os

import pretty_errors

from conv2d.torch_model_2d import Train as Train_2d
from fft_conv1d_lstm.torch_model_1d import Train as Train_1d

print('1: Conv2D model')
print('2: Conv1D_LSTM model')

model_choice = input("\nWhich model to train : ").strip()
try:
    model_choice = int(model_choice)
    if model_choice not in (1, 2):
        raise Exception
except:
    raise Exception("Input number 1 or 2")

epochs = input("Number of epochs to train : ").strip()
try:
    epochs = int(epochs)
except:
    raise Exception("Input number 1 or 2")

live_plot = input("Display live plot? [Y/n] : ").strip().lower()
if live_plot in {'n', 'no'}:
    live_plot = False
else:
    live_plot = True

curr_path = os.getcwd()

if model_choice == 1:
    print(f"\nTraining Conv2D model...")
    
    trainer = Train_2d(batch_size=16, val_batch_size=8, learning_rate=1e-3, start_new=True, liveplot=live_plot)
    trainer.train(num_epochs=epochs)

elif model_choice == 2:
    print(f"\nTraining Conv1D_LSTM model")
    
    trainer = Train_1d(batch_size=16, val_batch_size=8, learning_rate=1e-3, start_new=True, liveplot=live_plot)
    trainer.train(num_epochs=epochs)