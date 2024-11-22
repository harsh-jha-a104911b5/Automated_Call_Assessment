import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import os

def load_audio_data(csv_file_path):
    """
    Load audio features and labels from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file containing audio features and labels.

    Returns:
        tuple:
            pd.DataFrame: The full DataFrame (for saving later).
            np.ndarray: Feature vectors (X).
            np.ndarray or None: Labels (y), if available (treat 'unknown' as missing).
    """
    df = pd.read_csv(csv_file_path)

    # Extract numeric columns for features
    X = df.select_dtypes(include=[np.number])  # Only numeric columns
    y = df['Emotion'].replace('Unknown', np.nan) if 'Emotion' in df.columns else None

    return df, X.values, y

def generate_labeled_csv(csv_file_path, tess_model_path, output_csv_path):
    """
    Generate a labeled CSV file for the custom audio dataset using the pre-trained TESS model.

    Args:
        csv_file_path (str): Path to the CSV file with custom audio features.
        tess_model_path (str): Path to the pre-trained TESS model.
        output_csv_path (str): Path to save the labeled CSV file.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Load audio data
    full_data, audio_features, labels = load_audio_data(csv_file_path)

    # Load the pre-trained TESS model
    if not os.path.exists(tess_model_path):
        raise FileNotFoundError(f"TESS model not found at '{tess_model_path}'")
    tess_model = load(tess_model_path)

    # Check for missing labels
    if labels.isnull().any():
        print("Found 'unknown' labels. Generating pseudo-labels using the TESS model...")
        pseudo_labels = tess_model.predict(audio_features)
        # Replace only 'unknown' labels with predictions
        full_data.loc[labels.isnull(), 'Emotion'] = pseudo_labels
        print("Pseudo-labels generated successfully.")
    else:
        print("No 'unknown' labels found. Using existing labels.")

    # Save the labeled dataset to a new CSV file
    full_data.to_csv(output_csv_path, index=False)
    print(f"Labeled dataset saved to '{output_csv_path}'.")

# Example Usage
if __name__ == "__main__":
    # File paths
    custom_csv_path = 'D:/Final project/emotion_analysis_project/datasets/audio_dataset/custom_audio_features.csv'
    tess_model_path = 'D:/Final project/emotion_analysis_project/models/tess_random_forest_model.pkl'
    output_csv_path = 'D:/Final project/emotion_analysis_project/datasets/audio_dataset/custom_audio_labeled2.csv'

    # Generate labeled CSV
    generate_labeled_csv(custom_csv_path, tess_model_path, output_csv_path)
