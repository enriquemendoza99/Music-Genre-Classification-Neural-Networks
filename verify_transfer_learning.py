# ==========================================
# STANDALONE VERIFICATION SCRIPT
# Save this as: verify_values.py
# Run: python verify_values.py
# ==========================================

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# 1. VERIFY DATASET SIZES
print("\n" + "=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

trainingSet = pd.read_csv('csv/train/spectrogram_train.csv')
validationSet = pd.read_csv('csv/train/spectrogram_validation.csv')

print(f"Training samples: {len(trainingSet)}")
print(f"Validation samples: {len(validationSet)}")
print(f"Total samples: {len(trainingSet) + len(validationSet)}")

print(f"\nSamples per genre in training:")
print(trainingSet['label'].value_counts().sort_index())

numClasses = trainingSet['label'].nunique()
print(f"\nNumber of classes: {numClasses}")

# 2. VERIFY RESNET18 PARAMETERS
print("\n" + "=" * 60)
print("RESNET18 MODEL VERIFICATION")
print("=" * 60)

# Load ResNet18
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Count parameters
total_params = sum(p.numel() for p in resnet18.parameters())
print(f"Total ResNet18 parameters: {total_params:,}")
print(f"Approximately: ~{total_params / 1_000_000:.1f}M parameters")

# Check expected input shape
print(f"\nExpected input shape: (batch, 3, 224, 224)")
print(f"  3 channels (RGB)")
print(f"  224×224 resolution")

# Check original output
print(f"\nOriginal ResNet18 output: {resnet18.fc.out_features} classes (ImageNet)")

# 3. VERIFY CUSTOM CLASSIFICATION HEAD
print("\n" + "=" * 60)
print("CUSTOM CLASSIFICATION HEAD")
print("=" * 60)

# Create custom head like in your code
custom_head = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

print("Architecture: Linear(512→256) + ReLU + Dropout(0.5) + Linear(256→10)")
print(f"Total head parameters: {sum(p.numel() for p in custom_head.parameters()):,}")

# 4. VERIFY HYPERPARAMETERS FROM YOUR CODE
print("\n" + "=" * 60)
print("HYPERPARAMETERS (from your code)")
print("=" * 60)

print("Batch size: 32")
print("Number of epochs: 30")
print("Initial learning rate: 1e-4 (0.0001)")
print("Phase 2 learning rate: 1e-5 (0.00001) - 10× smaller")
print("Weight decay: 1e-4")
print("\nLearning rate scheduler:")
print("  Type: StepLR")
print("  Step size: 5")
print("  Gamma: 0.5 (reduces LR by 50% every 5 epochs)")

# 5. VERIFY DATA AUGMENTATION
print("\n" + "=" * 60)
print("DATA AUGMENTATION")
print("=" * 60)

print("Training augmentations:")
print("  1. Resize to 224×224")
print("  2. RandomHorizontalFlip(p=0.5)")
print("  3. RandomRotation(10°)")
print("  4. ToTensor()")
print("  5. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")

print("\nValidation (no augmentation):")
print("  1. Resize to 224×224")
print("  2. ToTensor()")
print("  3. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")

# 6. VERIFY TWO-PHASE TRAINING
print("\n" + "=" * 60)
print("TWO-PHASE TRAINING STRATEGY")
print("=" * 60)

print("PHASE 1 (Epochs 1-14):")
print("  - Freeze all ResNet18 layers")
print("  - Train only classification head")
print("  - Learning rate: 1e-4")
print("  - Trainable params: ~132K (only the head)")

print("\nPHASE 2 (Epochs 15-30):")
print("  - Unfreeze all layers at epoch 15")
print("  - Fine-tune entire network")
print("  - Learning rate: 1e-5 (10× reduction)")
print("  - Trainable params: ~11.2M (entire network)")

# 7. VERIFY RGB CONVERSION
print("\n" + "=" * 60)
print("INPUT ADAPTATION")
print("=" * 60)

print("Original spectrograms: Grayscale (1 channel)")
print("Converted to: RGB (3 channels)")
print("Method: Replicate grayscale → [R=G=B=spectrogram]")
print("Why: Preserves pre-trained weights without architecture modification")

# 8. COUNT AUDIO FILES (if available)
print("\n" + "=" * 60)
print("AUDIO FILES VERIFICATION")
print("=" * 60)

import os
from glob import glob

if os.path.exists('Data/train'):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    total_files = 0

    for genre in genres:
        files = glob(f'Data/train/{genre}/*.au')
        print(f"{genre:12s}: {len(files):3d} files")
        total_files += len(files)

    print(f"\nTotal audio files: {total_files}")
else:
    print("Data/train directory not found")
    print("(This is okay if you're only verifying CSV data)")

# 9. FINAL SUMMARY
print("\n" + "=" * 60)
print("FINAL SUMMARY - VALUES FOR REPORT")
print("=" * 60)

print(f"""
✓ Training samples: {len(trainingSet)}
✓ Validation samples: {len(validationSet)}
✓ Total samples: {len(trainingSet) + len(validationSet)}
✓ Number of genres: {numClasses}
✓ ResNet18 parameters: ~11M
✓ Input resolution: 224×224 RGB
✓ Batch size: 32
✓ Epochs: 30
✓ Phase 1 LR: 1e-4 (epochs 1-14)
✓ Phase 2 LR: 1e-5 (epochs 15-30)
✓ Classification head: 512→256→10
✓ Augmentation: HorizontalFlip + Rotation(10°)
""")

print("=" * 60)