from warnings import filterwarnings

import pretty_errors

from fft_conv1d_lstm.torch_model_1d import Predict as Predict_1d
from conv2d.torch_model_2d import Predict as Predict_2d

filterwarnings(action='ignore')

print('\n1: Conv1D_LSTM model')
print('2: Conv2D model')

model_choice = input("\nModel to use : ").strip()
try:
    model_choice = int(model_choice)
    if model_choice not in {1, 2}:
        raise Exception
except:
    raise Exception("Input number 1 or 2")

image_path = input("Path of to predict : ").strip()

plot_image = input("Display image? [Y/n] : ").strip().lower()
if plot_image in {'n', 'no'}:
    plot_image = False
else:
    plot_image = True

if model_choice == 1:    
    predictor = Predict_1d()
    predictor.predict(image_path, display_image=plot_image)

elif model_choice == 2:
    predictor = Predict_2d()
    predictor.predict(image_path, display_image=plot_image)