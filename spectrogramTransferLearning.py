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
import torchvision.models as models

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from kaggleTest import generate_kaggle_predictions_images
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class SpectrogramDataset(Dataset):
    def __init__(self, df, is_training=False):

        self.df = df.reset_index(drop=True)
        transforms = [T.Resize((224, 224))]
        #if is_training:
            #transforms += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(10)]
        transforms += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        return self.transform(image), int(row['label'])


class TransferLearningCNN(nn.Module):
    def __init__(self, numClasses):
        super(TransferLearningCNN, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, numClasses)
        )

    def forward(self, x):
        return self.model(x)

    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    # Defines Paths to Training and Validation CSVs
    training_csv_path = 'csv/train/spectrogram_train.csv'
    validation_csv_path = 'csv/train/spectrogram_validation.csv'

    # Define Hyperparameters
    batchSize = 32
    numEpochs = 50
    learningRate = 1e-4

    # Allow to use GPU if Available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)


    # Load Data
    print("\nLoading CSV Files...")
    trainingSet = pd.read_csv(training_csv_path)
    validationSet = pd.read_csv(validation_csv_path)
    numClasses = trainingSet['label'].nunique()

    print("Training Dataframe Shape:", trainingSet.shape)
    print("Validation Dataframe Shape:", validationSet.shape)
    print("Number of Classes:", numClasses)

    training_loader = DataLoader(SpectrogramDataset(trainingSet, True), batch_size=batchSize, shuffle=True)
    validation_loader = DataLoader(SpectrogramDataset(validationSet, False), batch_size=batchSize, shuffle=False)

    model = TransferLearningCNN(numClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningRate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("\nModel:")
    print(model)

    best_validation_accuracy = 0.0
    best_model_path = 'model/best_spectrogram_transfer00.pth'
    os.makedirs('model', exist_ok=True)

    print("\nTraining and Validating Model...")
    for epoch in range(numEpochs):
        print(f"\nEpoch {epoch + 1}/{numEpochs}")

        # Training all layers
        if epoch == 25:
            print("*** Unfreezing all layers ***")
            model.unfreeze_all_layers()
            optimizer = optim.Adam(model.parameters(), lr=learningRate / 10, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Train
        model.train()
        running_loss= 0.0
        correct= 0
        total = 0
        for inputs, labels in training_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
                inputs, labels = inputs.to(device), labels.to(device)
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

    # Final Evaluation
    print("\nLoading Best Model for Final Evaluation")
    best_model = TransferLearningCNN(numClasses).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.eval()

    allLabels = []
    allPredictions = []
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = outputs.max(1)
            allLabels.extend(labels.cpu().numpy())
            allPredictions.extend(predicted.cpu().numpy())

    allLabels = np.array(allLabels)
    allPredictions = np.array(allPredictions)

    print(f"\nFinal Validation Accuracy: {accuracy_score(allLabels, allPredictions):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(allLabels, allPredictions))
    print("\nClassification Report:")
    print(classification_report(allLabels, allPredictions, digits=4))

    cm = confusion_matrix(allLabels, allPredictions)
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0b0f1a')
    ax.set_facecolor('#111520')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')

    ax.set_title(
        f'Confusion Matrix — Transfer Learning (ResNet18)\nAccuracy: {accuracy_score(allLabels, allPredictions):.2%}',
        color='#e8eaf0', fontsize=12, fontweight='bold', pad=12
    )
    ax.tick_params(colors='#7a8099', labelsize=9)
    ax.xaxis.label.set_color('#7a8099')
    ax.yaxis.label.set_color('#7a8099')
    ax.spines[:].set_color('#1e2535')
    plt.xticks(rotation=45, ha='right')

    for text in disp.text_.ravel():
        text.set_color('white')
        text.set_fontsize(9)
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_transfer_learning.png', dpi=180,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print("Saved: plots/confusion_matrix_transfer_learning.png")