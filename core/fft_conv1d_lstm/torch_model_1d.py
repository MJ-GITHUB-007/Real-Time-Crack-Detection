import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pretty_errors

class FFTLayer(nn.Module):
    def __init__(self):
        super(FFTLayer, self).__init__()

    def forward(self, inputs):
        # Convert color images to grayscale
        grayscale_transform = transforms.Grayscale()
        inputs = torch.stack([grayscale_transform(inputs[i]) for i in range(inputs.size(0))])

        # Fourier transform
        fft_result = torch.fft.fftn(inputs)
        fft_shifted = torch.fft.fftshift(fft_result)
        log_magnitude = torch.log(torch.abs(fft_shifted) + 1e-8)

        flattened_log_magnitude = log_magnitude.view(log_magnitude.size(0), -1)
        expanded_log_magnitude = flattened_log_magnitude.unsqueeze(-1)

        return expanded_log_magnitude

class FFT_Conv1D_LSTM(nn.Module):
    def __init__(self):
        super(FFT_Conv1D_LSTM, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fft = FFTLayer()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1
        )
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.maxpool1 = nn.MaxPool1d(
            kernel_size=4,
            stride=2
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.maxpool2 = nn.MaxPool1d(
            kernel_size=4,
            stride=4
        )

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1
        )
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.maxpool3 = nn.MaxPool1d(
            kernel_size=4,
            stride=4
        )

        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1
        )
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.maxpool4 = nn.MaxPool1d(
            kernel_size=4,
            stride=4
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            bidirectional=False
        )

        self.fc1 = nn.Linear(
            in_features=1984,
            out_features=64
        )

        self.fc2 = nn.Linear(
            in_features=64,
            out_features=1
        )

    def forward(self, x):

        x = self.fft(x)

        x = x.permute(0, 2, 1)  # Adjusting dimensions for conv1d

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool4(x)

        x = x.permute(2, 0, 1)  # Adjusting dimensions for lstm
        
        x, _ = self.lstm(x)
        
        x = x.permute(1, 2, 0)  # Adjusting dimensions for flatten
        
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

class Train():
    def __init__(self, batch_size=32, val_batch_size=8, learning_rate=1e-3, start_new=True, liveplot=True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_path = os.getcwd()
        self.display = liveplot

        train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.train_dataset = datasets.ImageFolder(root=os.path.join(self.curr_path, 'data', 'train'), transform=train_transform)
        self.val_dataset = datasets.ImageFolder(root=os.path.join(self.curr_path, 'data', 'validation'), transform=val_transform)

        print(f"\nFound {len(self.train_dataset)} images for training")
        print(f"Found {len(self.val_dataset)} images for validating")

        if torch.cuda.is_available():
            print(f"Using GPU : {torch.cuda.get_device_name(0)}")
        else:
            print(f"No GPU available, using CPU.")

        # Move datasets to the device
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

        # Define your PyTorch model (make sure it's designed to run on the specified device)
        self.model = FFT_Conv1D_LSTM().to(self.device)
        if start_new:
            self.message = "new fft_conv1d_lstm_model"
        else:
            self.message = "existing fft_conv1d_lstm_model"
            try:
                self.model.load_state_dict(torch.load(os.path.join(self.curr_path, 'models', 'fft_conv1d_lstm_model.pth')))
            except:
                raise Exception(f"Model fft_conv1d_lstm_model failed to load")

        # Define loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7, weight_decay=False, amsgrad=False)

    def train(self, num_epochs) -> None:

        print(f"\nTraining {self.message} model for {num_epochs} epoch(s)...\n")

        if self.display:
            fig, axes = plt.subplots(1, 2, figsize=(9, 5))
            ax1, ax2 = axes

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):

            # Use tqdm for progress bar during training
            with tqdm(total=len(self.train_loader), bar_format=f'Epoch {epoch + 1}/{num_epochs} | Train      '+'|{bar:30}{r_bar}', unit=' batch(s)') as pbar:
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                    # Ensure labels are one-dimensional
                    labels = labels.unsqueeze(-1)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    self.optimizer.step()

                    # Update metrics
                    total_loss += loss.item()
                    predicted = (outputs.data > 0.5).float()
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    # Update progress bar
                    avg_loss = total_loss / total_samples
                    accuracy = correct_predictions / total_samples
                    pbar.set_postfix({'loss': avg_loss, 'accuracy': accuracy})
                    pbar.update(1)

                train_losses.append(total_loss / total_samples)
                train_accuracies.append(correct_predictions / total_samples)

                pbar.close()

            # Validation
            with torch.no_grad():
                # Use tqdm for progress bar during validation
                with tqdm(total=len(self.val_loader), bar_format=f'        '+' '*len(str(epoch+1))+' '*len(str(num_epochs))+'| Validation '+'|{bar:30}{r_bar}', unit=' batch(s)') as pbar:
                    val_total_loss = 0.0
                    val_correct_predictions = 0
                    val_total_samples = 0

                    for inputs, labels in self.val_loader:
                        inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                        # Ensure labels are one-dimensional
                        labels = labels.unsqueeze(-1)

                        # Forward pass for validation
                        outputs = self.model(inputs)
                        val_loss = self.criterion(outputs, labels)

                        # Update metrics for validation
                        val_total_loss += val_loss.item()
                        val_predicted = (outputs.data > 0.5).float()
                        val_total_samples += labels.size(0)
                        val_correct_predictions += (val_predicted == labels).sum().item()

                        # Update progress bar
                        val_avg_loss = val_total_loss / val_total_samples
                        val_accuracy = val_correct_predictions / val_total_samples
                        pbar.set_postfix({'val_loss': val_avg_loss, 'val_accuracy': val_accuracy})
                        pbar.update(1)

                    val_losses.append(val_total_loss / val_total_samples)
                    val_accuracies.append(val_correct_predictions / val_total_samples)

                    pbar.close()

            if self.display:
                # Update live plots
                ax1.clear()
                ax2.clear()

                # Adjust x-axis values
                x_values = list(range(1, epoch + 2))

                ax1.plot(x_values, train_losses, label='Training Loss')
                ax1.plot(x_values, val_losses, label='Validation Loss')
                ax1.set_title('Losses')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()

                ax2.plot(x_values, train_accuracies, label='Training Accuracy')
                ax2.plot(x_values, val_accuracies, label='Validation Accuracy')
                ax2.set_title('Accuracies')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()

                plt.suptitle('Live Training Stats\nfft_conv1d_lstm_model', fontsize=14)
                plt.pause(0.1)


            # Save the trained model
            self.criterion = nn.BCELoss()
            self.optimizer = optim.Adam(self.model.parameters())
            torch.save(self.model.state_dict(), os.path.join(self.curr_path, 'models', 'fft_conv1d_lstm_model.pth'))
            print()
        print(f'Training complete\n')
        if self.display:
            plt.show()

class Test():
    def __init__(self, batch_size=8) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_path = os.getcwd()

        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.test_dataset = datasets.ImageFolder(root=os.path.join(self.curr_path, 'data', 'test'), transform=test_transform)

        print(f"\nFound {len(self.test_dataset)} images for testing")

        if torch.cuda.is_available():
            print(f"Using GPU : {torch.cuda.get_device_name(0)}")
        else:
            print(f"No GPU available, using CPU.")

        # Move dataset to the device
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        # Define your PyTorch model (make sure it's designed to run on the specified device)
        self.model = FFT_Conv1D_LSTM().to(self.device)
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.curr_path, 'models', 'fft_conv1d_lstm_model.pth')))
        except:
            raise Exception(f"Model fft_conv1d_lstm_model failed to load")

        self.criterion = nn.BCELoss()

    def test(self, con_matrix=True) -> None:

        print(f"\nEvaluating fft_conv1d_lstm_model model...\n")

        # Use tqdm for progress bar during testing
        with torch.no_grad():
            all_labels = []
            all_predictions = []

            with tqdm(total=len(self.test_loader), bar_format='Evaluation '+'|{bar:30}{r_bar}', unit=' batch(s)') as pbar:
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                    labels = labels.unsqueeze(-1)

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # Update metrics
                    total_loss += loss.item()
                    predicted = (outputs.data > 0.5).float()
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    # Collect labels and predictions for later evaluation
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    # Update progress bar
                    avg_loss = total_loss / total_samples
                    accuracy = correct_predictions / total_samples
                    pbar.set_postfix({'loss': avg_loss, 'accuracy': accuracy})
                    pbar.update(1)

                pbar.close()

            # Convert lists to numpy arrays
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

        print(f'\nEvaluation complete')

        # Generate and print classification report and confusion matrix
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive']))

        if con_matrix:
            con_mat = confusion_matrix(all_labels, all_predictions)
            display = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=['Negative', 'Positive'])
            display.plot()
            plt.title('Conv1D_LSTM model')
            plt.show()

class Predict():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_path = os.getcwd()

        if torch.cuda.is_available():
            print(f"\nUsing GPU : {torch.cuda.get_device_name(0)}")
        else:
            print(f"\nNo GPU available, using CPU.")

        # Define your PyTorch model (make sure it's designed to run on the specified device)
        self.model = FFT_Conv1D_LSTM().to(self.device)
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.curr_path, 'models', 'fft_conv1d_lstm_model.pth')))
        except:
            raise Exception(f"Model fft_conv1d_lstm_model failed to load")

    def predict(self, image_path, display_image=True):
        predict_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = predict_transform(image).unsqueeze(0).to(self.device)

        # Perform prediction
        with torch.no_grad():
            output = self.model(input_tensor).cpu()
            output = np.array(output)[0][0]
            output = float(output)

            if output > 0.5:
                prediction = 'Negative'
                confidence = str(round(output*100, 4)) + '%'
            else:
                prediction = 'Positive'
                confidence = str(round(1 - output*100, 4)) + '%'
        
        # Print result
        print(f"\nPredicted given image \"{image_path}\" as \"{prediction}\" with {confidence} confidence using Conv1D_LSTM model.\n")

        # Display the original image and the prediction
        if display_image:
            plt.imshow(image)
            plt.title(f'Prediction: {prediction}\nConfidence: {confidence}', fontsize=16)
            plt.xlabel(image_path, fontsize=13)
            plt.xticks([])
            plt.yticks([])  
            plt.show()

if __name__ == '__main__':
    # trainer = Train(batch_size=16, val_batch_size=8, learning_rate=1e-3, start_new=True, liveplot=False)
    # trainer.train(num_epochs=25)

    # tester = Test(batch_size=8)
    # tester.test()

    predictor = Predict()
    image_path = 'data/test/Negative/00254.jpg'
    predictor.predict(image_path, display_image=True)