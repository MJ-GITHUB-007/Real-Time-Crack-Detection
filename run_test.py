from warnings import filterwarnings

import pretty_errors

from fft_conv1d_lstm.torch_model_1d import Test as Test_1d
from conv2d.torch_model_2d import Test as Test_2d

filterwarnings(action='ignore')

print('\n1: Conv1D_LSTM model')
print('2: Conv2D model')

model_choice = input("\nModel to evaluate : ").strip()
try:
    model_choice = int(model_choice)
    if model_choice not in {1, 2}:
        raise Exception
except:
    raise Exception("Input number 1 or 2")

plot_matrix = input("Display Confusion Matrix? [Y/n] : ").strip().lower()
if plot_matrix in {'n', 'no'}:
    plot_matrix = False
else:
    plot_matrix = True

if model_choice == 1:    
    tester = Test_1d(batch_size=8)
    tester.test(con_matrix=plot_matrix)

elif model_choice == 2:
    tester = Test_2d(batch_size=8)
    tester.test(con_matrix=plot_matrix)