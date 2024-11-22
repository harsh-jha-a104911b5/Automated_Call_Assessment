import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

def load_and_train_model(csv_file_path, model_save_path='models/emotion_model.pkl'):
    """
    Train an emotion detection model using labeled audio data from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file with audio features and labels.
        model_save_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Load the labeled dataset
    data = pd.read_csv(csv_file_path)

    # Extract features (X) and labels (y)
    X = data.select_dtypes(include=[np.number])  # Assuming all numeric columns are features
    y = data['Emotion']  # Labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    # File path for the labeled dataset
    labeled_csv_path = 'D:/Final project/emotion_analysis_project/datasets/audio_dataset/custom_audio_labeled2.csv'
    model_save_path = 'models/emotion_detection_model.pkl'

    # Train the model using the labeled data
    load_and_train_model(csv_file_path=labeled_csv_path, model_save_path=model_save_path)
