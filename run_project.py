from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors
import os

print('1: Conv2D model')
print('2: Conv1D_LSTM model')
prompt = input("Which model to train : ")

try:
    prompt = int(prompt)
except:
    raise Exception("Input number 1 or 2")

curr_path = os.getcwd()

if prompt == 1:
    print(f"Running python {os.path.join(curr_path, 'conv2d', 'train.py')}\n")
    os.system(f"python {os.path.join(curr_path, 'conv2d', 'train.py')}")
elif prompt == 2:
    print(f"Running python {os.path.join(curr_path, 'conv1d_lstm', 'train.py')}\n")
    os.system(f"python {os.path.join(curr_path, 'conv1d_lstm', 'train.py')}")
else:
    raise Exception("Input number 1 or 2")