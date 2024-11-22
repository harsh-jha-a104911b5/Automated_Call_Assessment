import os
import json
import torch
import librosa
from pyannote.audio import Pipeline
import whisper
from transformers import pipeline
from tqdm import tqdm
from joblib import load
import numpy as np

# Load models
print("Loading diarization model...")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")

print("Loading transcription model...")
whisper_model = whisper.load_model("base")

print("Loading sentiment analysis model...")
sentiment_analyzer = pipeline("sentiment-analysis")

print("Loading emotion detection model...")
emotion_model_path = 'models/emotion_detection_model.pkl'
emotion_detection_model = load(emotion_model_path)

# Output folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

def load_audio_without_ffmpeg(audio_path, target_sample_rate=16000):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != target_sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
    return y

def extract_audio_features(audio_file_path):
    audio, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)
    features = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean, zero_crossing_rate_mean, rms_mean])
    return features

def perform_diarization(audio_path):
    diarization = diarization_pipeline(audio_path)
    segments = []
    for segment, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="Performing diarization", unit="segment"):
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
        })
    return segments

def perform_transcription_with_speaker_labels(audio_path, diarization_segments):
    audio = load_audio_without_ffmpeg(audio_path)
    transcription_result = whisper_model.transcribe(audio, word_timestamps=True)
    whisper_segments = transcription_result["segments"]

    results = []
    for diarized_segment in tqdm(diarization_segments, desc="Performing transcription", unit="segment"):
        speaker = diarized_segment["speaker"]
        start_time = diarized_segment["start"]
        end_time = diarized_segment["end"]

        segment_text = ""
        for whisper_segment in whisper_segments:
            whisper_start = whisper_segment["start"]
            whisper_end = whisper_segment["end"]
            if whisper_start < end_time and whisper_end > start_time:
                segment_text += whisper_segment["text"] + " "

        results.append({
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "text": segment_text.strip(),
        })
    return results

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

def analyze_emotion(audio_path):
    features = extract_audio_features(audio_path)
    emotion = emotion_detection_model.predict([features])[0]
    return emotion

def combine_results_with_emotion(audio_path, transcription_results):
    combined_results = []
    for entry in transcription_results:
        speaker = entry["speaker"]
        text = entry["text"]
        sentiment_label, sentiment_score = analyze_sentiment(text)
        emotion = analyze_emotion(audio_path)

        combined_results.append({
            "speaker": speaker,
            "start": entry["start"],
            "end": entry["end"],
            "text": text,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment_score
            },
            "emotion": emotion
        })
    return combined_results

def save_results_to_json(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    audio_path = input("Enter the path to the audio file: ").strip()
    print("Performing diarization...")
    diarization_segments = perform_diarization(audio_path)
    
    print("Performing transcription...")
    transcription_results = perform_transcription_with_speaker_labels(audio_path, diarization_segments)

    print("Combining results with emotion and sentiment analysis...")
    combined_results = combine_results_with_emotion(audio_path, transcription_results)

    output_path = os.path.join(output_folder, f"{os.path.basename(audio_path).split('.')[0]}_analysis.json")
    save_results_to_json(combined_results, output_path)

    print(f"Analysis results saved to {output_path}")

if __name__ == "__main__":
    main()
