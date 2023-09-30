from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors
import os

print('1: Conv2D model')
print('2: Conv1D_LSTM model')
prompt = input("Which model to test : ")

try:
    prompt = int(prompt)
except:
    raise Exception("Input number 1 or 2")

curr_path = os.getcwd()

if prompt == 1:
    print(f"Running test Conv2D model\n")
    os.system(f"python {os.path.join(curr_path, 'conv2d', 'test.py')}")
elif prompt == 2:
    print(f"Running test Conv1D_LSTM model\n")
    os.system(f"python {os.path.join(curr_path, 'conv1d_lstm', 'test.py')}")
else:
    raise Exception("Input number 1 or 2")