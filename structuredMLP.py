# Import essential libraries
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from glob import glob

import librosa
import librosa.display

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



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

# Defines MLP Architecture
class MLP(nn.Module):
    def __init__(self, inputSize, numClasses):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(inputSize, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(64, numClasses)
        )

    def forward(self, x):
        return self.network(x)
  

if __name__ == '__main__':
    # DATA PROCESSING
    # PATH TO TRAINING DATASET IS DEFINED HERE
    # Loads files for each genre
    blues_audio = glob('Data/train/blues/*.au')
    classical_audio = glob('Data/train/classical/*.au')
    country_audio = glob('Data/train/country/*.au')
    disco_audio = glob('Data/train/disco/*.au')
    hiphop_audio = glob('Data/train/hiphop/*.au')
    jazz_audio = glob('Data/train/jazz/*.au')
    metal_audio = glob('Data/train/metal/*.au')
    pop_audio = glob('Data/train/pop/*.au')
    reggae_audio = glob('Data/train/reggae/*.au')
    rock_audio = glob('Data/train/rock/*.au')


    # Create a features list
    features = []

    # Defines number of Mel-Frequency Cepstral Coefficients to extract
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
    numEpochs = 30
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

    # Allow to use GPU if Available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device:", device)
    
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


    # Training and Validation
    print("\nTraining and Validating Model...")

    for epoch in range(numEpochs):
        print("\nEpoch", epoch + 1, "/", numEpochs)

        # Trains MLP
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            for inputs, labels in val_loader:
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

        print("\nRunning MLP Model Training and Validation Stats:")
        print("Training Loss: {:.4f}".format(training_loss))
        print("Training Accuracy: {:.4f}".format(training_accuracy))
        print("Validation Loss: {:.4f}".format(validation_loss))
        print("Validation Accuracy: {:.4f}".format(validation_accuracy))

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("\nNew Best MLP Model Saved!")
            print("Best Validation Accuracy: {:.4f}".format(best_validation_accuracy))


    # Loads Best Model for Evaluation
    print("\nLoading Best MLP for Final Evaluation on Validation Set")
    best_model = MLP(inputSize=inputSize, numClasses=numClasses).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.eval()

    allLabels = []
    allPredictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
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
    print("\nFinal Validation Accuracy of Best MLP Model: {:.4f}".format(final_validation_accuracy))

    # Generates Confusion Matrix
    MLPcm = confusion_matrix(allLabels, allPredictions)
    print("\nMLP Confusion Matrix:")
    print(MLPcm)

    # Generates Classification Report
    print("MLP Classification Report:")
    print(classification_report(allLabels, allPredictions, digits=4))
