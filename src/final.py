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
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

class ConversationAnalyzer:
    def __init__(self):
        # Initialize paths
        self.models_dir = Path("models")
        self.output_folder = Path("results")
        self.output_folder.mkdir(exist_ok=True)

        # Load environment variables
        load_dotenv()

        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models
        self._load_models()
        
        # Set up Gemini
        self._setup_gemini()

    def _load_models(self):
        """Load all required models with GPU support"""
        print("Loading diarization model...")
        # Retrieve Hugging Face token from environment variable
        huggingface_token = os.getenv('HUGGINGFACE_AUTH_TOKEN')
        if not huggingface_token:
            raise ValueError("HUGGINGFACE_AUTH_TOKEN not found in .env file")

        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=huggingface_token
        )
        # Move diarization pipeline to GPU
        self.diarization_pipeline.to(self.device)

        print("Loading transcription model...")
        # Load Whisper model with CUDA support
        self.whisper_model = whisper.load_model("base", device=self.device)

        print("Loading sentiment analysis model...")
        # Load sentiment analysis pipeline with GPU support
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            device=0 if torch.cuda.is_available() else -1,
            batch_size=32  # Adjust based on your GPU memory
        )

        print("Loading emotion detection model...")
        emotion_model_path = self.models_dir / 'emotion_detection_model.pkl'
        self.emotion_detection_model = load(emotion_model_path)

    def _setup_gemini(self):
        """Set up the Gemini model"""
        # Retrieve Gemini API key from environment variable
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    def load_audio(self, audio_path, target_sample_rate=16000):
        """Load audio file with GPU acceleration for processing"""
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != target_sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
        
        # Convert to torch tensor and move to GPU if available
        if torch.cuda.is_available():
            return torch.from_numpy(y).to(self.device)
        return y

    def extract_audio_features(self, audio_file_path):
        """Extract audio features with GPU acceleration where possible"""
        audio, sr = librosa.load(audio_file_path, sr=None)
        
        # Convert to torch tensor for GPU processing
        if torch.cuda.is_available():
            audio_tensor = torch.from_numpy(audio).to(self.device)
            # Convert back to numpy for librosa processing
            audio = audio_tensor.cpu().numpy()

        # Extract features
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
        
        features = np.hstack([mfccs_mean, chroma_mean, spectral_contrast_mean, 
                            zero_crossing_rate_mean, rms_mean])
        return features

    def perform_diarization(self, audio_path):
        """Perform speaker diarization with GPU acceleration"""
        diarization = self.diarization_pipeline(audio_path)
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
            })
        return segments

    def perform_transcription(self, audio_path, diarization_segments):
        """Perform transcription with speaker labels using GPU"""
        audio = self.load_audio(audio_path)
        
        # Use GPU for transcription
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        transcription_result = self.whisper_model.transcribe(audio, word_timestamps=True)
        whisper_segments = transcription_result["segments"]

        results = []
        for diarized_segment in diarization_segments:
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

    def analyze_sentiment(self, texts):
        """Analyze sentiment with batch processing on GPU"""
        if not isinstance(texts, list):
            texts = [texts]
            
        results = self.sentiment_analyzer(texts, batch_size=32)
        
        if len(texts) == 1:
            return results[0]['label'], results[0]['score']
        return results

    def analyze_emotion(self, audio_path):
        """Analyze emotion from audio with GPU acceleration where possible"""
        features = self.extract_audio_features(audio_path)
        
        # Convert to torch tensor for GPU processing if model supports it
        if hasattr(self.emotion_detection_model, 'predict_proba') and torch.cuda.is_available():
            features_tensor = torch.from_numpy(features).float().to(self.device)
            emotion = self.emotion_detection_model.predict(features_tensor.cpu().numpy().reshape(1, -1))[0]
        else:
            emotion = self.emotion_detection_model.predict([features])[0]
            
        return emotion

    def combine_results(self, audio_path, transcription_results):
        """Combine all analysis results with batch processing"""
        # Batch process sentiments for all texts
        texts = [entry["text"] for entry in transcription_results]
        sentiment_results = self.analyze_sentiment(texts)
        
        combined_results = []
        for i, entry in enumerate(transcription_results):
            emotion = self.analyze_emotion(audio_path)
            combined_results.append({
                "speaker": entry["speaker"],
                "start": entry["start"],
                "end": entry["end"],
                "text": entry["text"],
                "sentiment": {
                    "label": sentiment_results[i]['label'],
                    "score": sentiment_results[i]['score']
                },
                "emotion": emotion
            })
        return combined_results

    def generate_summary(self, data):
        """Generate summary using Gemini model"""
        conversation_text, speaker_data = self._parse_conversation_data(data)
        prompt = self._construct_prompt(conversation_text, speaker_data)

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error during model inference: {str(e)}"

    def _parse_conversation_data(self, data):
        """Parse conversation data for summary generation"""
        conversation_text = ""
        speaker_data = {
            "SPEAKER_00": {"emotions": [], "sentiments": [], "text": []},
            "SPEAKER_01": {"emotions": [], "sentiments": [], "text": []},
        }

        for item in data:
            speaker = item['speaker']
            text = item['text']
            sentiment = item['sentiment']['label']
            emotion = item['emotion']

            conversation_text += f"{speaker}: {text}\n"

            if speaker in speaker_data:
                speaker_data[speaker]["emotions"].append(emotion)
                speaker_data[speaker]["sentiments"].append(sentiment)
                speaker_data[speaker]["text"].append(text)

        for speaker, details in speaker_data.items():
            details["emotions"] = ", ".join(set(details["emotions"])) or "No emotions detected."
            details["sentiments"] = ", ".join(set(details["sentiments"])) or "No sentiments detected."
            details["text"] = " ".join(details["text"])

        return conversation_text, speaker_data

    def _construct_prompt(self, conversation_text, speaker_data):
        """Construct prompt for summary generation"""
        return f"""
        You are an advanced conversational analysis model. Please analyze the following conversation and provide a detailed and structured summary, incorporating insights into the dynamics, emotions, and sentiments of the speakers.

        1. Main Topic(s):
           - Identify the primary topics discussed in the conversation. Summarize the overall subject matter and the specific themes explored.

        2. Dynamics of the Conversation:
           - Analyze the flow of the conversation. How do the speakers interact with each other? Are there shifts in tone, interruptions, or specific conversational patterns? Provide insights on the overall structure and progression of the dialogue.

        3. Speaker Analysis:
           - For each speaker, provide a detailed breakdown:
             - Key Points: Highlight the main ideas or contributions made by the speaker.
             - Predominant Emotions: Identify the primary emotions conveyed by the speaker, including any shifts or variations throughout the conversation.
             - Sentiments: Summarize the positive, negative, or neutral sentiments expressed by the speaker.
             - Behavior Analysis: What is the speaker's conversational style (e.g., assertive, passive, polite, confrontational)? Does the speaker express empathy, frustration, or any other notable behaviors?
             - Tone: Describe the tone used by the speaker (e.g., formal, casual, sarcastic, enthusiastic).
             
           - Example:
             - SPEAKER_00:
               Key Points: [Your answer here]
               Predominant Emotions: {speaker_data['SPEAKER_00']['emotions']}
               Sentiments: {speaker_data['SPEAKER_00']['sentiments']}
               Behavior Analysis: [Your answer here]
               Tone: [Your answer here]
             
             - SPEAKER_01:
               Key Points: [Your answer here]
               Predominant Emotions: {speaker_data['SPEAKER_01']['emotions']}
               Sentiments: {speaker_data['SPEAKER_01']['sentiments']}
               Behavior Analysis: [Your answer here]
               Tone: [Your answer here]

        4. Emotional Tone of the Conversation:
           - Identify the overall emotional tone of the conversation.

        5. Key Insights and Learnings:
           - Provide a deeper analysis of the conversation.

        7. Conversation Context:
           - {conversation_text}
        """

    def analyze_audio(self, audio_path):
        """Main function to analyze audio file with GPU acceleration"""
        try:
            # Perform diarization
            print("Performing diarization...")
            diarization_segments = self.perform_diarization(audio_path)

            # Perform transcription with speaker labels
            print("Performing transcription...")
            transcription_results = self.perform_transcription(audio_path, diarization_segments)

            # Batch process sentiments for better GPU utilization
            print("Analyzing sentiments and emotions...")
            texts = [segment["text"] for segment in transcription_results]
            sentiment_results = self.analyze_sentiment(texts)
            
            # Combine results with emotion analysis
            print("Combining results...")
            combined_results = []
            for i, entry in enumerate(transcription_results):
                emotion = self.analyze_emotion(audio_path)
                combined_results.append({
                    "speaker": entry["speaker"],
                    "start": entry["start"],
                    "end": entry["end"],
                    "text": entry["text"],
                    "sentiment": {
                        "label": sentiment_results[i]['label'],
                        "score": sentiment_results[i]['score']
                    },
                    "emotion": emotion
                })

            # Generate summary
            print("Generating summary...")
            summary = self.generate_summary(combined_results)

            # Save summary
            output_file = self.output_folder / "summary.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)

            torch.cuda.empty_cache()  # Clear GPU memory after processing
            return summary

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    """Command line interface for the analyzer"""
    audio_path = input("Enter the path to the audio file: ").strip()
    analyzer = ConversationAnalyzer()
    summary = analyzer.analyze_audio(audio_path)
    print(f"\nAnalysis complete. Summary saved to: {analyzer.output_folder}/summary.txt")

if __name__ == "__main__":
    main()