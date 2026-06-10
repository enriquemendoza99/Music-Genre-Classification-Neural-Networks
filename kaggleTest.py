# Import Essential Libraries
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from spectrogramExtraction import compute_log_spectrogram, save_spectrogram_png

def generate_kaggle_predictions_images(best_Model, imgHeight=224, imgWidth=224, channels='RGB'):
    encoder = LabelEncoder() 

    genres = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
    ]

    encoder.fit(genres)

    # Define Data Paths and Spectrogram Paths
    test_audio_root = 'data/test'
    spectrogram_root = 'spectrograms/test'
    
    # Same transform as training
    transform = T.Compose([
        T.Resize((imgHeight, imgWidth)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # for RGB images
        # T.Normalize(mean=[0.5], std=[0.5]) --- for grayscale images
    ])

    # Gathers Training Files
    pattern = os.path.join(test_audio_root, '*.au')
    files = glob(pattern)
    images = []

    # Generates Spectrogram for Each File
    for filePath in files:
        # Defines Output Path for Spectrogram
        base = os.path.splitext(os.path.basename(filePath))[0]
        outPath = os.path.join(spectrogram_root, base + '.png')
        images.append(outPath)

        # Skip if spectrogram already exists
        if os.path.exists(outPath):
            print(f"Skipping (already exists): {outPath}")
            continue

        print("Processing File: ", filePath)

        spectrogram_db = compute_log_spectrogram(filePath)

        save_spectrogram_png(spectrogram_db, outPath)

    print("Completed Generating Spectrograms for Test Data")

    # Predict Kaggle Test Data
    print("Predicting from Kaggle Test Data...")
    # Allow to use GPU if Available
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    best_Model.eval()

    predictions = []

    with torch.no_grad():
        for img_path in images:
            # Load and transform image
            image = Image.open(img_path).convert(channels) # Use 'L' for CNN, 'RGB' for Transfer Learning
            image = transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            
            # Get prediction
            output = best_Model(image)
            _, predicted = output.max(1)
            predictions.append(predicted.item())
    
    y_final_pred = np.array(predictions)

    # Creates CSV for Kaggle Submission
    print("Creating Kaggle CSV")
    test_filenames = [os.path.basename(item) for item in files]
    kaggle_test_predictions = pd.DataFrame({
        'id': test_filenames, 
        'class': encoder.inverse_transform(y_final_pred)
    })
    kaggle_test_predictions.to_csv('TheOutlierDetectivesPredictions.csv', index=False)
    print("Kaggle Test Complete")


""" Use for structured data version
# Custom Normalization Tool (I wasn't sure if we could use sklearn for this)
class Normalize:
    def fit(self, X):
        self.means = X.mean()
        self.stds = X.std()
        return self
    def transform(self, X):
        return (X - self.means) / self.stds
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
# Custom Scaling Tool (I wasn't sure if we could use sklearn for this)
class Scaler:
    def fit(self, X):
        self.mins = X.min()
        self.maxs = X.max()
        return self
    def transform(self, X):
        return (X - self.mins) / (self.maxs - self.mins)
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def generate_kaggle_predictions_structuredData(bestModel, num_mfcc=13):
    # PREDICTING ON KAGGLE TEST DATA
    # PATH TO KAGGLE TEST DATA IS DEFINED HERE
    # Loads files for Kaggle test data
    kaggle_test_audio = glob('Data/test/*.au')

    # Creates Kaggle Test Features
    kaggle_test_features = []

    # Load Kaggle test data into Librosa and Extract Features
    print("\n\nProcessing Kaggle Test Data...")
    for file in kaggle_test_audio:
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
        kaggle_test_features.append({
                        **{f'mfcc_{i+1}': mfcc_mean[i] for i in range(len(mfcc_mean))},
                        **{f'stft_{i+1}':stft_features[i] for i in range(len(stft_features))},
                        **{f'chroma_{i+1}': chroma_mean[i] for i in range(len(chroma_mean))},
                        **{f'chromaCQT_{i+1}': chroma_cqt_mean[i] for i in range(len(chroma_cqt_mean))},
                        **{f'chromaCENS_{i+1}': chroma_cens_mean[i] for i in range(len(chroma_cens_mean))},
                        **{f'spec_contrast_{i+1}': spec_contrast_mean[i] for i in range(len(spec_contrast_mean))},
                        'zcr': zcr_mean,
                        'rms': rms_mean
                        })
        
    encoder = LabelEncoder()

    # Creates a dataframe from the extracted features
    kaggle_test_df = pd.DataFrame(kaggle_test_features)

    # Normalizes Kaggle Test (features) dataset
    normalizer = Normalize()
    kaggle_test_norm = normalizer.transform(kaggle_test_df)

    # Scales Kaggle Test (features) dataset
    scaler = Scaler()
    kaggle_test_norm_scale = scaler.transform(kaggle_test_norm)

    # Performs PCA on Kaggle Test (features) dataset
    pca = PCA(n_components=0.95)
    kaggle_test_norm_scale_pca = pca.transform(kaggle_test_norm_scale)
    X_test = pd.DataFrame(kaggle_test_norm_scale_pca)

    # Predict Kaggle Test Data
    print("Predicting from Kaggle Test Data...")
    y_final_pred = bestModel.predict(X_test)

    # Creates CSV for Kaggle Submission
    print("Creating Kaggle CSV")
    kaggle_test_audio = [item.split('/')[-1] for item in kaggle_test_audio]
    kaggle_test_predictions = pd.DataFrame({'id': kaggle_test_audio, 'class': encoder.inverse_transform(y_final_pred)})
    kaggle_test_predictions.to_csv('TheOutlierDetectivesPredictions.csv', index=False)
    print("Kaggle Test Complete")
"""