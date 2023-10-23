from warnings import filterwarnings

import pretty_errors

from fft_conv1d_lstm.torch_model_1d import Train as Train_1d
from conv2d.torch_model_2d import Train as Train_2d

filterwarnings(action='ignore')

print('\n1: Conv1D_LSTM model')
print('2: Conv2D model')

model_choice = input("\nModel to train : ").strip()
try:
    model_choice = int(model_choice)
    if model_choice not in {1, 2}:
        raise Exception
except:
    raise Exception("Input number 1 or 2")

epochs = input("Number of epochs to train : ").strip()
try:
    epochs = int(epochs)
except:
    raise Exception("Input number 1 or 2")

existing_model = input("Train existing model? [Y/n] : ").strip().lower()
if existing_model in {'n', 'no'}:
    new_model = True
else:
    new_model = False

live_plot = input("Display live plot? [Y/n] : ").strip().lower()
if live_plot in {'n', 'no'}:
    live_plot = False
else:
    live_plot = True

if model_choice == 1:    
    trainer = Train_1d(batch_size=16, val_batch_size=8, learning_rate=1e-6, start_new=new_model, liveplot=live_plot)
    trainer.train(num_epochs=epochs)

elif model_choice == 2:
    trainer = Train_2d(batch_size=16, val_batch_size=8, learning_rate=1e-6, start_new=new_model, liveplot=live_plot)
    trainer.train(num_epochs=epochs)