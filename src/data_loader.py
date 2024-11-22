# data_loader.py

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

def load_audio_files(directory):
    """
    Load all audio files from the directory for the custom dataset.
    
    Args:
        directory (str): Path to the custom audio dataset folder.
        
    Returns:
        List[Tuple[str, np.ndarray]]: List of tuples with file paths and audio data.
    """
    audio_data = []
    
    # Loop through each audio file in the directory
    for file_name in tqdm(os.listdir(directory), desc="Loading custom audio files", unit="file"):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                audio_data.append((file_path, audio))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return audio_data

def load_audio_files_from_folders(directory):
    """
    Load all audio files from the emotion folders in the TESS dataset.
    
    Args:
        directory (str): Path to the root directory of TESS dataset containing emotion subfolders.
        
    Returns:
        List[Tuple[str, np.ndarray, str]]: List of tuples with file paths, audio data, and emotion label.
    """
    audio_data = []
    
    # Loop through each emotion folder (e.g., 'anger', 'fear', 'happy', etc.)
    for emotion_folder in tqdm(os.listdir(directory), desc="Loading TESS emotion folders", unit="folder"):
        emotion_folder_path = os.path.join(directory, emotion_folder)
        
        if os.path.isdir(emotion_folder_path):
            # Loop through each WAV file in the emotion folder
            for file_name in os.listdir(emotion_folder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(emotion_folder_path, file_name)
                    try:
                        audio, sr = librosa.load(file_path, sr=None)
                        audio_data.append((file_path, audio, emotion_folder))  # Include emotion as label
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    
    return audio_data

def extract_audio_features(audio_data):
    """
    Extract features from a list of loaded audio files.
    
    Args:
        audio_data (List[Tuple[str, np.ndarray]]): List of tuples with file paths and audio data (for custom dataset).
        audio_data (List[Tuple[str, np.ndarray, str]]): List of tuples with file paths, audio data, and emotion labels (for TESS dataset).
        
    Returns:
        pd.DataFrame: DataFrame containing feature vectors, filenames, and emotion labels.
    """
    features = []
    file_names = []
    emotions = []
    
    for file_path, audio, *emotion in tqdm(audio_data, desc="Extracting features", unit="file"):
        try:
            # Feature extraction using librosa
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=16000).T, axis=0)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=16000).T, axis=0)
            zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
            rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
            
            # Combine features into a single feature vector
            feature_vector = np.hstack([mfccs, chroma, spectral_contrast, zero_crossings, rms])
            features.append(feature_vector)
            file_names.append(os.path.basename(file_path))
            if emotion:
                emotions.append(emotion[0])  # Add emotion label for TESS dataset
            else:
                emotions.append("Unknown")  # Label for custom dataset (no emotion folder structure)
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
    
    # Create a DataFrame with named columns for features and filenames
    feature_names = [f'mfcc_{i+1}' for i in range(13)] + \
                    [f'chroma_{i+1}' for i in range(12)] + \
                    [f'spectral_contrast_{i+1}' for i in range(7)] + \
                    [f'zero_crossings_{i+1}' for i in range(1)] + \
                    [f'rms_{i+1}' for i in range(1)]
    
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['Filename'] = file_names
    features_df['Emotion'] = emotions  # Add emotion column
    
    return features_df

def save_features_to_csv(features_df, csv_path):
    """
    Save extracted features to a CSV file.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing extracted features and labels.
        csv_path (str): Path to save the CSV file.
        
    Returns:
        None
    """
    features_df.to_csv(csv_path, index=False)

# Main function to perform data loading and feature extraction for both datasets
if __name__ == "__main__":
    # Get the paths of the datasets from the user
    custom_audio_dataset_path = input("Enter the path for your custom audio dataset folder: ")
    tess_audio_dataset_path = input("Enter the path for the TESS dataset root folder: ")
    
    print("Loading audio data for custom audio dataset...")
    # Load custom audio dataset audio files
    custom_audio_data = load_audio_files(custom_audio_dataset_path)
    
    print("Extracting features for custom audio dataset...")
    # Extract features from the custom audio dataset
    custom_features_df = extract_audio_features(custom_audio_data)
    
    # Save extracted features to a CSV file for your custom audio dataset
    custom_csv_path = os.path.join(custom_audio_dataset_path, 'custom_audio_features.csv')
    print(f"Saving features to {custom_csv_path}...")
    save_features_to_csv(custom_features_df, custom_csv_path)
    print(f"Features for custom audio dataset saved to {custom_csv_path}")
    
    print("Loading audio data for TESS dataset...")
    # Load TESS dataset audio files
    tess_audio_data = load_audio_files_from_folders(tess_audio_dataset_path)
    
    print("Extracting features for TESS dataset...")
    # Extract features from the TESS dataset
    tess_features_df = extract_audio_features(tess_audio_data)
    
    # Save extracted features to a CSV file for the TESS dataset
    tess_csv_path = os.path.join(tess_audio_dataset_path, 'tess_features.csv')
    print(f"Saving features to {tess_csv_path}...")
    save_features_to_csv(tess_features_df, tess_csv_path)
    print(f"Features for TESS dataset saved to {tess_csv_path}")
