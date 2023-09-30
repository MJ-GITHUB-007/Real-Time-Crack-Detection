from warnings import filterwarnings
filterwarnings(action='ignore')
import pretty_errors

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

curr_path = os.getcwd()

class Test():
    def __init__(self, model_path=os.path.join(curr_path, 'models/conv2D_model.keras')) -> None:
        self.model = load_model(model_path)
        self.data_path = os.path.join(curr_path, 'data', 'test')
        self.actual, self.prediction = None, None

        self.report = None
        self.confmat = None
        self.display = None
    
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
    test = Test()
    test.do_test()
    test.show_eval()