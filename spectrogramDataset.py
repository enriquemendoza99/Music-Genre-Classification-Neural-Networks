# Import Essential Libraries
import os
from glob import glob

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Define Data Paths and CSV Paths
spectrogram_root = 'spectrograms/train'
csv_output_directory = 'csv/train'

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

testSize = 0.2
randomState = 42


# Builds a Empty List for All Files and Genres
dataList = []

# Fills List with Files and Genres
for genre in genres:
    # Defines Path to Files
    pattern = os.path.join(spectrogram_root, genre, '*.png')
    files = glob(pattern)

    print("Found", len(files), "files for genre:", genre)

    # Appends Files and Genres into List
    for filePath in files:
        dataList.append({
            'filepath': filePath,
            'genre': genre
        })

# Creates a Pandas Dataframe
df = pd.DataFrame(dataList)

print("\nFull Dataframe Generated:")
print(df.head())
print(df.shape)


# Encodes Genre Labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['genre'])

print("\nDataframe with Numerical Labels:")
print(df.head())


# Create Training and Validation Datasets
trainSet, validSet = train_test_split(df, test_size=testSize, stratify=df['label'], random_state=randomState)
print("\nTraining and Validation Sets Created")
print("Training Set Shape:", trainSet.shape)
print("Validation Set Shape:", validSet.shape)


# Save Training and Validation Dataframes to CSVs
# Ensure Path Exists or Create Path
if not os.path.exists(csv_output_directory):
    os.makedirs(csv_output_directory)

# Defines Path for CSVs
training_csv_path = os.path.join(csv_output_directory, 'spectrogram_train.csv')
validation_csv_path = os.path.join(csv_output_directory, 'spectrogram_validation.csv')

# Creates CSVs from Training and Validation Dataframes
trainSet.to_csv(training_csv_path, index=False)
validSet.to_csv(validation_csv_path, index=False)

print("\nSaved Training and Validation Set CSVs")
print("Training CSV:", training_csv_path)
print("Validation CSV:", validation_csv_path)


