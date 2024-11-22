import librosa
import numpy as np
from joblib import load
import os
import pandas as pd

# Function to extract audio features (MFCCs, Chroma, Spectral Contrast, Zero-Crossing Rate)
def extract_audio_features(audio_file_path):
    """
    Extract audio features from the given audio file.
    Args:
        audio_file_path (str): Path to the audio file.
    Returns:
        np.ndarray: Extracted features (e.g., MFCC, Chroma, Spectral Contrast, etc.).
    """
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_file_path, sr=None)  # `sr=None` preserves the original sample rate
    print(f"Audio file '{audio_file_path}' loaded. Sample rate: {sr}, Audio length: {len(audio)} samples.")
    
    # Extract MFCC (Mel-frequency cepstral coefficients) from the audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)  # Get the mean of the MFCCs over time

    # Extract Chroma (Pitch Class) features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Extract Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)

    # Extract RMS (Root Mean Square) energy
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)

    # Concatenate all features into a single feature vector
    features = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean, zero_crossing_rate_mean, rms_mean])
    
    print("Extracted features:", features.shape)  # Check the shape of extracted features (should be 34)

    return features

# Function to test emotion detection model
def test_emotion_detection_model(audio_file_path, model_path='models/emotion_detection_model.pkl'):
    """
    Test the emotion detection model on a given audio file.
    
    Args:
        audio_file_path (str): Path to the audio file.
        model_path (str): Path to the trained emotion detection model.
        
    Returns:
        str: Predicted emotion.
    """
    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    print(f"Audio file found: {audio_file_path}")

    # Extract features from the input audio file
    audio_features = extract_audio_features(audio_file_path)
    
    # Load the trained emotion detection model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load(model_path)
    print(f"Model loaded from {model_path}")

    # Make prediction using the trained model
    predicted_emotion = model.predict([audio_features])  # Pass the feature as a 2D array
    print("Prediction complete.")

    # Return the predicted emotion
    return predicted_emotion[0]

# Example Usage
if __name__ == "__main__":
    # Input audio file (WAV file)
    audio_file_path = input("Enter the path of the audio file to test: ")

    # Path to the trained model (update as per your model's path)
    model_path = 'models/emotion_detection_model.pkl'
    
    try:
        # Predict the emotion of the input audio file
        predicted_emotion = test_emotion_detection_model(audio_file_path, model_path)
        print(f"Predicted Emotion: {predicted_emotion}")
    except Exception as e:
        print(f"Error: {e}")
