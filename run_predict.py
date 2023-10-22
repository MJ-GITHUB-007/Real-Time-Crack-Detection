from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from fft_conv1d_lstm.keras_model_1d import FFTLayer

curr_path = os.getcwd()

class Prediction():
    def __init__(self, model_name: str = None, image_path: os.path.normpath = None) -> None:
        model_path=os.path.join(curr_path, 'models', model_name)
        
        if model_name == 'conv1D_lstm_model.keras':
            self.model = load_model(model_path, custom_objects={'FFTLayer': FFTLayer})
        elif model_name == 'conv2D_model.keras':
            self.model = load_model(model_path)
    
    def __read_predictions(self, predictions):
        return_predictions = []
        for prediction in predictions:
            prediction = prediction[0]
            if prediction < 0.5:
                return_predictions.append(0)
            else:
                return_predictions.append(1)
        return return_predictions
    
    def do_test(self, num_epochs=1000) -> None:
        neg_predictions = []
        pos_predictions = []

        print(f'\nGetting predictions')
        print(f'Please wait...')

        for image_class in ('Positive', 'Negative'):
            images = os.listdir(os.path.join(self.data_path, image_class))
            for image in images:
                image = cv2.imread(filename=os.path.join(self.data_path, image_class, image))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.divide(image, 255)
                image = np.array([image])

                if image_class == 'Positive':
                    pos_predictions.append(self.model.predict(image, verbose=0))
                elif image_class == 'Negative':
                    neg_predictions.append(self.model.predict(image, verbose=0))

        neg_predictions = self.__read_predictions(neg_predictions)
        pos_predictions = self.__read_predictions(pos_predictions)

        neg_actual = [0] * len(neg_predictions)
        pos_actual = [1] * len(pos_predictions)

        self.actual = neg_actual + pos_actual
        self.predictions = neg_predictions + pos_predictions
        print(f'Done')
    
    def show_eval(self):
        self.report = classification_report(y_true=self.actual, y_pred=self.predictions, target_names=['Negative', 'Positive'])
        print(f'\n\nClassification Report : \n{self.report}')

        self.conf_mat = confusion_matrix(y_true=self.actual, y_pred=self.predictions)
        self.display = ConfusionMatrixDisplay(confusion_matrix=self.conf_mat, display_labels=['Negative', 'Positive'])
        self.display.plot()
        plt.show()

if __name__ == '__main__':
    image_path = input(f'Input absolute path of image to predict :\n')
    path = os.path.normpath(image_path)

    print('\n1: Conv2D model')
    print('2: Conv1D_LSTM model')
    prompt = input("Which model to use : ")

    try:
        prompt = int(prompt)
    except:
        raise Exception("Input number 1 or 2")

    curr_path = os.getcwd()

    if prompt == 1:
        print(f"Running test Conv2D model\n")
        model_name = 'conv2D_model.keras'
    elif prompt == 2:
        print(f"Running test Conv1D_LSTM model\n")
        model_name = 'conv1D_lstm_model.keras'
    else:
        raise Exception("Input number 1 or 2")

    test = Prediction(model_name=model_name, image_path=image_path)
    test.do_test()
    test.show_eval()