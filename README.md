# Music Genre Classification — MLP, CNN, and Transfer Learning
Music genre classification system comparing three neural network architectures — MLP on structured audio features, CNN on spectrogram images, and Transfer Learning with ResNet18 — trained on the GTZAN dataset across 10 genres using PyTorch.

## How to Run

1. Create a virtual environment: python -m venv venv
2. Activate it: venv\Scripts\activate (Windows) or source venv/bin/activate (Mac/Linux)
3. Install dependencies: pip install -r requirements.txt
4. Download the dataset and place the genre folders inside data/train/
5. Run scripts in this order:
   python spectrogramExtraction.py
   python spectrogramDataset.py
   python modelAnalysis.py

## File Manifest

1. spectrogramExtraction.py — Loads audio files and generates log-scale power spectrogram PNG images for each genre.
2. spectrogramDataset.py — Builds training and validation CSV files from the generated spectrogram images.
3. structuredMLP.py — Defines the MLP model architecture and custom normalization and scaling tools.
4. spectrogramCNN.py — Defines the CNN model architecture trained on grayscale spectrogram images.
5. spectrogramTransferLearning.py — Defines the Transfer Learning model using pre-trained ResNet18 with two-phase training.
6. modelAnalysis.py — Main executable. Runs grid search, trains all three models, evaluates performance, and generates comparison plots.
7. kaggleTest.py — Generates Kaggle test predictions from the best trained models.
7. requirements.txt — All required packages and versions.

## Results

| Model | Best Validation Accuracy |
|---|---|
| MLP | 71.11% |
| CNN | 61.67% |
| Transfer Learning (ResNet18) | **70.56%** |
