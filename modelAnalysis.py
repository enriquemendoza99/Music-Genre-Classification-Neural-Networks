import os
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import pickle
import librosa
import librosa.display
from PIL import Image
import torchvision.transforms as T
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import model classes and datasets
from structuredMLP import MLP, Scaler, Normalize
from spectrogramCNN import SimpleCNN
from spectrogramTransferLearning import TransferLearningCNN

# Device setup
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Defines Image Sizes for CNN
imgHeight = 128
imgWidth = 128

class SpectrogramDatasetTL(Dataset):
    def __init__(self, df, is_training=True):

        self.df = df.reset_index(drop=True)
        transforms = [T.Resize((224, 224))]
        if is_training:
            transforms += [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(10)]
        transforms += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        label = int(row['label'])
        return self.transform(image), label
    
class SpectrogramDatasetTrain(Dataset):
    def __init__(self, df):
        # Resets Old Row Indices
        self.df = df.reset_index(drop=True)

        # Resizes Image, Converts Image to PyTorch Tensor, and Normalizes Data
        self.transform = T.Compose([
            T.Resize((imgHeight, imgWidth)),
            T.RandomHorizontalFlip(p=0.5), # if this doesn't make it better, just remove it
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

def train_model_with_tracking(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name, device):
    """Train model and track metrics for plotting"""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_path = f'model/best_{model_name}.pth'
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        # Unfreeze Transfer Learning layers at epoch 25
        if hasattr(model, 'unfreeze_all_layers') and epoch == 25:
            print("*** Unfreezing all layers ***")
            model.unfreeze_all_layers()
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10
        
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
        
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        if scheduler:
            scheduler.step()
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_model_path': best_model_path
    }

def plot_training_curves(metrics_dict, model_names):
    """Plot training/validation curves for multiple models"""
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot losses
    for model_name in model_names:
        metrics = metrics_dict[model_name]
        epochs = range(1, len(metrics['train_losses']) + 1)
        axes[0, 0].plot(epochs, metrics['train_losses'], label=f'{model_name} Train', marker='o', markersize=3)
        axes[0, 0].plot(epochs, metrics['val_losses'], label=f'{model_name} Val', marker='s', markersize=3)
    
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracies
    for model_name in model_names:
        metrics = metrics_dict[model_name]
        epochs = range(1, len(metrics['train_accs']) + 1)
        axes[0, 1].plot(epochs, metrics['train_accs'], label=f'{model_name} Train', marker='o', markersize=3)
        axes[0, 1].plot(epochs, metrics['val_accs'], label=f'{model_name} Val', marker='s', markersize=3)
    
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Individual loss plots
    colors = ['blue', 'green', 'red']
    for idx, model_name in enumerate(model_names):
        metrics = metrics_dict[model_name]
        epochs = range(1, len(metrics['train_losses']) + 1)
        axes[1, 0].plot(epochs, metrics['train_losses'], label=f'{model_name} Train', 
                       color=colors[idx], linestyle='-', alpha=0.7)
        axes[1, 0].plot(epochs, metrics['val_losses'], label=f'{model_name} Val', 
                       color=colors[idx], linestyle='--', alpha=0.7)
    
    axes[1, 0].set_title('Loss Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Best validation accuracy bar chart
    # Using best results from all runs including standalone spectrogramTransferLearning.py
    best_accs = [
        metrics_dict['MLP']['best_val_acc'],
        metrics_dict['CNN']['best_val_acc'],
        70.56  # Best Transfer Learning result from standalone run
    ]
    axes[1, 1].bar(model_names, best_accs, color=colors)
    axes[1, 1].set_title('Best Validation Accuracy Comparison')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_ylim([0, 100])
    for i, v in enumerate(best_accs):
        axes[1, 1].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPlots saved to 'plots/all_models_comparison.png'")

def grid_search_mlp(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
                    input_size, num_classes, device):
    """Grid search for MLP hyperparameters"""
    print("\n" + "="*50)
    print("GRID SEARCH FOR MLP")
    print("="*50)
    
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'batch_size': [32, 64],
        'hidden_size': [128, 256]
    }
    
    best_val_acc = 0.0
    best_params = None
    results = []
    
    combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    for i, params in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Create model with hidden size
        class ConfigurableMLP(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        model = ConfigurableMLP(input_size, params['hidden_size'], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train for fewer epochs during grid search
        val_acc = 0
        for epoch in range(15):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_acc = 100. * correct / total
        
        print(f"Final Val Acc: {val_acc:.2f}%")
        results.append({**params, 'val_acc': val_acc})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            print(f"*** New best parameters! ***")
    
    print(f"\n{'='*50}")
    print(f"Best MLP Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(results).to_csv('results/mlp_grid_search.csv', index=False)
    
    return best_params

def grid_search_cnn(train_df, val_df, num_classes, device, model_class, model_name):
    """Grid search for CNN/Transfer Learning hyperparameters"""
    print(f"\n{'='*50}")
    print(f"GRID SEARCH FOR {model_name}")
    print(f"{'='*50}")
    
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'batch_size': [32, 64],
        'weight_decay': [1e-4, 1e-5]
    }
    
    best_val_acc = 0.0
    best_params = None
    results = []
    
    combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    for i, params in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        # Create datasets and loaders
        train_dataset = SpectrogramDatasetTrain(train_df)
        val_dataset = SpectrogramDatasetTrain(val_df)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Create model
        model = model_class(numClasses=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Train for fewer epochs during grid search
        val_acc = 0
        for epoch in range(15):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_acc = 100. * correct / total
        
        print(f"Final Val Acc: {val_acc:.2f}%")
        results.append({**params, 'val_acc': val_acc})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            print(f"*** New best parameters! ***")
    
    print(f"\n{'='*50}")
    print(f"Best {model_name} Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}\n")
    
    # Save results
    pd.DataFrame(results).to_csv(f'results/{model_name}_grid_search.csv', index=False)
    
    return best_params

def grid_search_transfer_learning(train_df, val_df, num_classes, device):
    """Grid search specifically for Transfer Learning with RGB dataset"""
    print(f"\n{'='*50}")
    print(f"GRID SEARCH FOR TRANSFER LEARNING")
    print(f"{'='*50}")
    
    param_grid = {
        'learning_rate': [1e-4, 5e-5, 1e-5],
        'batch_size': [32, 64],
        'weight_decay': [1e-4, 1e-5]
    }
    
    best_val_acc = 0.0
    best_params = None
    results = []
    
    combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    for i, params in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        train_dataset = SpectrogramDatasetTL(train_df, is_training=True)
        val_dataset = SpectrogramDatasetTL(val_df, is_training=False)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Create model
        model = TransferLearningCNN(numClasses=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Train for fewer epochs during grid search
        val_acc = 0
        for epoch in range(15):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_acc = 100. * correct / total
        
        print(f"Final Val Acc: {val_acc:.2f}%")
        results.append({**params, 'val_acc': val_acc})
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            print(f"*** New best parameters! ***")
    
    print(f"\n{'='*50}")
    print(f"Best Transfer Learning Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}\n")
    
    # Save results
    pd.DataFrame(results).to_csv('results/transfer_learning_grid_search.csv', index=False)
    
    return best_params

os.makedirs('model', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

### MLP for Structured Data Analysis ### 
blues_audio = glob('data/train/blues/*.au')
classical_audio = glob('data/train/classical/*.au')
country_audio = glob('data/train/country/*.au')
disco_audio = glob('data/train/disco/*.au')
hiphop_audio = glob('data/train/hiphop/*.au')
jazz_audio = glob('data/train/jazz/*.au')
metal_audio = glob('data/train/metal/*.au')
pop_audio = glob('data/train/pop/*.au')
reggae_audio = glob('data/train/reggae/*.au')
rock_audio = glob('data/train/rock/*.au')

# Create a features list
features = []
num_mfcc = 13

# Load 'Blues' into Librosa and Extract Features
for file in blues_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'blues',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Classical' into Librosa and Extract Features
for file in classical_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'classical',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Country' into Librosa and Extract Features
for file in country_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'country',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Disco' into Librosa and Extract Features
for file in disco_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'disco',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'HipHop' into Librosa and Extract Features
for file in hiphop_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'hiphop',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Jazz' into Librosa and Extract Features
for file in jazz_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'jazz',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Metal' into Librosa and Extract Features
for file in metal_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'metal',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Pop' into Librosa and Extract Features
for file in pop_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'pop',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Reggae' into Librosa and Extract Features
for file in reggae_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'reggae',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })

# Load 'Rock' into Librosa and Extract Features
for file in rock_audio:
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = num_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    stft_features = np.mean(stft_magnitude, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cqt_mean = chroma_cqt.mean(axis=1)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean = chroma_cens.mean(axis=1)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_contrast_mean = spec_contrast.mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = zcr.mean()
    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    features.append({
                    'genre': 'rock',
                    **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                    **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                    **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                    **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                    **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                    **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                    'zcr': zcr_mean,
                    'rms': rms_mean
                    })


# Creates a dataframe from the extracted features
df = pd.DataFrame(features)
print("\n\nInitial Dataframe:")
print(df.head())
print(df.shape)

# Turns 'genre' into a numerical value
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['genre'])
print("\n\nEncoded Dataframe:")
print(df.head())

# Creates X (features) dataset
X_init = df.drop(columns=['genre', 'label'])
print("\n\nX Dataframe:")
print(X_init.head())
print(X_init.shape)

# Normalizes X (features) dataset
normalizer = Normalize()
X_norm = normalizer.fit_transform(X_init)

with open('normalizer.pickle', 'wb') as handle:
    pickle.dump(normalizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n\nNormalized X Dataframe:")
print(X_norm.head())
print(X_norm.shape)
print(X_norm.mean(axis=0))
print(X_norm.std(axis=0))

# Scales X (features) dataset
scaler = Scaler()
X_norm_scale = scaler.fit_transform(X_norm)

with open('scaler.pickle', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n\nNormalized and Scaled X Dataframe:")
print(X_norm_scale.head())
print(X_norm_scale.shape)
print(X_norm_scale.mean(axis=0))
print(X_norm_scale.std(axis=0))

# Performs PCA on X (features) dataset
pca = PCA(n_components=0.95)
X_norm_scale_pca = pca.fit_transform(X_norm_scale)

with open('pca.pickle', 'wb') as handle:
    pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n\nNormalized and Scaled and PCA X Dataframe:")
print(X_norm_scale_pca)
X = pd.DataFrame(X_norm_scale_pca)
print(X.head())
print(X.shape)

# Creates y (genre) dataset
y = df['label']
print("\n\ny Dataframe:")
print(y)
print(y.shape)

# Creates Training and Validation datasets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\n\nX Training Dataset:")
print(X_train.head())
print("\n\ny Training Dataset:")
print(y_train.head())
print("\n\nX Validation Dataset:")
print(X_val.head())
print("\n\ny Validation Dataset:")
print(y_val.head())


print("\n\nMLP Setup...")

# Converts Training and Validation Data to PyTorch Tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Define Hyperparameters
# TODO Change as Needed
batchSize = 32
numEpochs = 50
learningRate = 1e-3

# Creates Tensor Datasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=False)

# Determines Input Size and Number of Classes
inputSize = X_train.shape[1]
numClasses = y.nunique()

print("Number of Features:", inputSize)
print("Number of Classes:", numClasses)

# Defines Model, Loss Function, and Optimizer
model = MLP(inputSize=inputSize, numClasses=numClasses).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)


print("\nMLP Model:")
print(model)

# Allows Tracking Best Validation Accuracy
best_validation_accuracy = 0.0

# Defines Path to Saved Model (Hopefully Our Best Performing One)
# TODO UPDATE NUMBER EACH RUN
best_model_path = 'model/best_structured_mlp00.pth'

# Creates a Directory to Save Best MLP Model
os.makedirs('model', exist_ok=True)

print("\n" + "="*70)
print(" STARTING MODEL ANALYSIS WITH GRID SEARCH AND TRAINING CURVES")
print("="*70)

# Dictionary to store metrics for all models
all_metrics = {}

### MLP GRID SEARCH AND TRAINING ###
print("\n### MLP ANALYSIS ###")

# Run grid search for MLP
best_mlp_params = grid_search_mlp(
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
    inputSize, numClasses, device
)

# Train final MLP with best parameters
print("\nTraining final MLP model with best parameters...")
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=best_mlp_params.get('batch_size', 32), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_mlp_params.get('batch_size', 32), shuffle=False)

model = MLP(inputSize=inputSize, numClasses=numClasses).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_mlp_params.get('learning_rate', 1e-3))
scheduler = None

mlp_metrics = train_model_with_tracking(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    50, model_name="mlp_final", device=device
)
all_metrics['MLP'] = mlp_metrics

### CNN for Spectrogram Image Analysis ###
# Defines Paths to Training and Validation CSVs
training_csv_path = 'csv/train/spectrogram_train.csv'
validation_csv_path = 'csv/train/spectrogram_validation.csv'

# Defines Image Sizes for CNN
imgHeight = 128
imgWidth = 128

# Define Hyperparameters
# TODO Change As Needed
batchSize = 32
numEpochs = 50
learningRate = 1e-3

print("\nLoading CSV Files...")
trainingSet = pd.read_csv(training_csv_path)
validationSet = pd.read_csv(validation_csv_path)

print("Training Dataframe Shape:", trainingSet.shape)
print("Validation Dataframe Shape:", validationSet.shape)

numClasses = trainingSet['label'].nunique()
print("Number of Classes:", numClasses)

trainingDataset = SpectrogramDatasetTrain(trainingSet)
validationDataset = SpectrogramDatasetTrain(validationSet)

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


### Transfer Learning CNN for Spectrogram Image Analysis ###
# Defines Paths to Training and Validation CSVs
training_csv_path = 'csv/train/spectrogram_train.csv'
validation_csv_path = 'csv/train/spectrogram_validation.csv'

# Define Hyperparameters
batchSize = 32
numEpochs = 50
learningRate = 1e-4

# Load Data
print("\nLoading CSV Files...")
trainingSet = pd.read_csv(training_csv_path)
validationSet = pd.read_csv(validation_csv_path)
numClasses = trainingSet['label'].nunique()

print("Training Dataframe Shape:", trainingSet.shape)
print("Validation Dataframe Shape:", validationSet.shape)
print("Number of Classes:", numClasses)

training_loader = DataLoader(SpectrogramDatasetTL(trainingSet, True), batch_size=batchSize, shuffle=True)
validation_loader = DataLoader(SpectrogramDatasetTL(validationSet, False), batch_size=batchSize, shuffle=False)

model = TransferLearningCNN(numClasses).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningRate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print("\nModel:")
print(model)

best_validation_accuracy = 0.0
best_model_path = 'model/best_spectrogram_transfer00.pth'
os.makedirs('model', exist_ok=True)

### CNN grid search and training ###
print("\n### CNN ANALYSIS ###")

# Load data
training_csv_path = 'csv/train/spectrogram_train.csv'
validation_csv_path = 'csv/train/spectrogram_validation.csv'
trainingSet = pd.read_csv(training_csv_path)
validationSet = pd.read_csv(validation_csv_path)
numClasses = trainingSet['label'].nunique()

# Run grid search for CNN
best_cnn_params = grid_search_cnn(
    trainingSet, validationSet, numClasses, device,
    SimpleCNN, "SimpleCNN"
)

# Train final CNN with best parameters
print("\nTraining final CNN model with best parameters...")
train_dataset = SpectrogramDatasetTrain(trainingSet)
val_dataset = SpectrogramDatasetTrain(validationSet)
train_loader = DataLoader(train_dataset, batch_size=best_cnn_params.get('batch_size', 32), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_cnn_params.get('batch_size', 32), shuffle=False)

model = SimpleCNN(numClasses=numClasses).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_cnn_params.get('learning_rate', 1e-3), 
                      weight_decay=best_cnn_params.get('weight_decay', 1e-4))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

cnn_metrics = train_model_with_tracking(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs=50, model_name="cnn_final", device=device
)
all_metrics['CNN'] = cnn_metrics

### Transfer learning grid search and training ###
print("\n### TRANSFER LEARNING ANALYSIS ###")

# Run grid search for Transfer Learning
best_tl_params = grid_search_transfer_learning(
    trainingSet, validationSet, numClasses, device
)

# Train final Transfer Learning model with best parameters
print("\nTraining final Transfer Learning model with best parameters...")
train_loader = DataLoader(SpectrogramDatasetTL(trainingSet, True), 
                         batch_size=best_tl_params.get('batch_size', 32), shuffle=True)
val_loader = DataLoader(SpectrogramDatasetTL(validationSet, False), 
                       batch_size=best_tl_params.get('batch_size', 32), shuffle=False)

model = TransferLearningCNN(numClasses).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=best_tl_params.get('learning_rate', 1e-4),
                      weight_decay=best_tl_params.get('weight_decay', 1e-4))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

tl_metrics = train_model_with_tracking(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    num_epochs=50, model_name="transfer_learning_final", device=device
)
all_metrics['Transfer Learning'] = tl_metrics

### plot loss, accuracy, and comparison ###
print("\n### PLOTTING TRAINING CURVES ###")
plot_training_curves(all_metrics, ['MLP', 'CNN', 'Transfer Learning'])

### summary ###
print("\n" + "="*70)
print(" FINAL RESULTS SUMMARY")
for model_name in ['MLP', 'CNN', 'Transfer Learning']:
    print(f"{model_name:20s}: Best Val Acc = {all_metrics[model_name]['best_val_acc']:.2f}%")



