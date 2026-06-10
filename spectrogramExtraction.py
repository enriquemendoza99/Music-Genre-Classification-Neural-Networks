# Import Essential Libraries
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display


# Define Data Paths and Spectrogram Paths
train_audio_root = 'data/train'
spectrogram_root = 'spectrograms/train'

frameSize = 2048
hopSize = 512

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


# Computes Log Spectogram
def compute_log_spectrogram(filePath):

    # Loads Files into Librosa
    y, sr = librosa.load(filePath, sr=None)

    # Obtains the Power Spectrogram
    stft = librosa.stft(y, n_fft=frameSize, hop_length=hopSize)
    power_spectrogram = np.abs(stft) ** 2

    # Converts to Log Scale (db)
    spectrogram_db = librosa.power_to_db(power_spectrogram, ref=np.max)

    return spectrogram_db


# Saves Spectrogram PNG for Use in Models
def save_spectrogram_png(spectrogram_db, outPath):

    # Defines Output Directory/Path
    outDirectory = os.path.dirname(outPath)
    if not os.path.exists(outDirectory):
        os.makedirs(outDirectory)
    
    # Plots Spectrograms
    plt.figure(figsize=(3,3))
    librosa.display.specshow(spectrogram_db, y_axis='log', x_axis='time')
    plt.axis('off')
    plt.tight_layout(pad=0)

    # Saves Spectrogram Plots to Defined Directory/Path
    plt.savefig(outPath, bbox_inches='tight', pad_inches=0)
    plt.close()


# Loop Over All Genres and Files for Spectrogram Generation
for genre in genres:

    # Gathers Training Files
    pattern = os.path.join(train_audio_root, genre, '*.au')
    files = glob(pattern)

    # Generates Spectrogram for Each File
    for filePath in files:
        # Defines Output Path for Spectrogram
        base = os.path.splitext(os.path.basename(filePath))[0]
        outPath = os.path.join(spectrogram_root, genre, base + '.png')

        # Skip if spectrogram already exists
        if os.path.exists(outPath):
            print(f"Skipping (already exists): {outPath}")
            continue

        print("Processing File: ", filePath)

        spectrogram_db = compute_log_spectrogram(filePath)

        save_spectrogram_png(spectrogram_db, outPath)

print("Completed Generating Spectrograms for All Genres")


