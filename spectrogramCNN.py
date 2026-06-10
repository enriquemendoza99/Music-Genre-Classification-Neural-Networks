# Imports Essential Libraries
import os

import pandas as pd
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    
class SpectrogramDataset(Dataset):
    def __init__(self, df):
        # Resets Old Row Indices
        self.df = df.reset_index(drop=True)

        # Resizes Image, Converts Image to PyTorch Tensor, and Normalizes Data
        self.transform = T.Compose([
            T.Resize((imgHeight, imgWidth)),
            #T.RandomHorizontalFlip(p=0.5), # if this doesn't make it better, just remove it
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgPath = row['filepath']
        label = int(row['label'])

        # Opens Image
        image = Image.open(imgPath)

        # Converts Image to Grayscale
        image = image.convert('L')

        # Apply Transformations to Image
        image = self.transform(image)

        return image, label
    

class SimpleCNN(nn.Module):
    def __init__(self, numClasses):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flattened_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),  # for 3 convulution layers
            #nn.Linear(32 * 32 * 32, 128), # for 2 convulution layers
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, numClasses)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.flattened_layers(x)
        return x
    
if __name__ == '__main__':
    # Defines Paths to Training and Validation CSVs
    training_csv_path = 'csv/train/spectrogram_train.csv'
    validation_csv_path = 'csv/train/spectrogram_validation.csv'

    # Defines Image Sizes for CNN
    imgHeight = 128
    imgWidth = 128

    # Define Hyperparameters
    # TODO Change As Needed
    batchSize = 32
    numEpochs = 30
    learningRate = 1e-3


    # Allow to use GPU if Available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)

    print("\nLoading CSV Files...")
    trainingSet = pd.read_csv(training_csv_path)
    validationSet = pd.read_csv(validation_csv_path)

    print("Training Dataframe Shape:", trainingSet.shape)
    print("Validation Dataframe Shape:", validationSet.shape)

    numClasses = trainingSet['label'].nunique()
    print("Number of Classes:", numClasses)

    trainingDataset = SpectrogramDataset(trainingSet)
    validationDataset = SpectrogramDataset(validationSet)

    training_loader = DataLoader(trainingDataset, batch_size=batchSize, shuffle=True)
    validation_loader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False)


    model = SimpleCNN(numClasses=numClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


    print("\nModel:")
    print(model)

    # Allows Tracking Best Validation Accuracy
    best_validation_accuracy = 0.0

    # Defines Path to Saved Model (Hopefully Our Best Performing One)
    # TODO UPDATE NUMBER EACH RUN
    best_model_path = 'model/best_spectrogram_cnn00.pth'

    # Creates a Directory to Save Best CNN Model
    os.makedirs('model', exist_ok=True)

    print("\nTraining and Validating Model...")
    for epoch in range(numEpochs):
        print("\nEpoch", epoch + 1, "/", numEpochs)

        # Trains CNN
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        training_loss = running_loss / total
        training_accuracy = correct / total
        

        # Test on Validation Set
        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                validation_total += labels.size(0)
                validation_correct += predicted.eq(labels).sum().item()
        
        validation_loss = validation_loss / validation_total
        validation_accuracy = validation_correct / validation_total

        print("\nRunning Model Training and Validation Stats:")
        print("Training Loss: {:.4f}".format(training_loss))
        print("Training Accuracy: {:.4f}".format(training_accuracy))
        print("Validation Loss: {:.4f}".format(validation_loss))
        print("Validation Accuracy: {:.4f}".format(validation_accuracy))

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("\nNew Best Model Saved!")
            print("Best Validation Accuracy: {:.4f}".format(best_validation_accuracy))

        scheduler.step()


    print("\nTraining and Validation Complete")


    # Loads Best Model for Evaluation
    print("\nLoading Best for Final Evaluation on Validation Set")
    best_model = SimpleCNN(numClasses=numClasses).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.eval()

    allLabels = []
    allPredictions = []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = best_model(inputs)
            _, predicted = outputs.max(1)

            allLabels.extend(labels.cpu().numpy())
            allPredictions.extend(predicted.cpu().numpy())

    allLabels = np.array(allLabels)
    allPredictions = np.array(allPredictions)

    # Determines Overall Accuracy
    final_validation_accuracy = accuracy_score(allLabels, allPredictions)
    print("\nFinal Validation Accuracy of Best Model: {:.4f}".format(final_validation_accuracy))

    # Generates Confusion Matrix
    cm = confusion_matrix(allLabels, allPredictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Generates Classification Report
    print("Classification Report:")
    print(classification_report(allLabels, allPredictions, digits=4))


# Generates Kaggle Predictions
# generate_kaggle_predictions_images(best_model)