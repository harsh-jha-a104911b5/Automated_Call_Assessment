# Conversation Analysis Platform 🎙️

An advanced audio analysis platform that leverages AI to provide deep insights into conversations. This tool performs speaker diarization, transcription, sentiment analysis, and emotion detection to generate comprehensive conversation summaries.

## Features 🚀

- **Speaker Diarization**: Automatically identifies and separates different speakers
- **Speech-to-Text**: Accurate transcription of conversations
- **Sentiment Analysis**: Analyzes the sentiment of each speaker's utterances
- **Emotion Detection**: Identifies emotions in speech using audio features
- **Conversation Summary**: AI-powered detailed analysis of conversation dynamics
- **Batch Processing**: Analyze multiple audio files simultaneously
- **GPU Acceleration**: Optimized performance with CUDA support

## Key Components ⚙️

- Single file analysis interface
- Batch processing capability (up to 10 files)
- Detailed conversation insights
- Downloadable analysis results
- Progress tracking and status updates

## Installation 🛠️

1. Clone the repository
```bash
git clone https://github.com/Sarthakischill/Conversation_Analysis.git
cd conversation-analysis-platform
```

2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Set up authentication
- Create a Hugging Face account
- Accept the license for pyannote/speaker-diarization-3.1
- Get your Hugging Face token
- Replace the token in `final.py`
```python
# In final.py
self.diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="your-token-here"
)
```

## Usage 💡

1. Start the application
```bash
streamlit run Home.py
```

2. Access the web interface at `http://localhost:8501`

3. Choose between:
   - Single file analysis (Home page)
   - Batch analysis (Batch Analysis page)

4. Upload WAV format audio file(s)

5. Click "Analyze Conversation" to start processing

6. View and download results

## Project Structure 📁
```
app/
├── pages/
│   └── 1_Batch_Analysis.py
├── models/
│   └── emotion_detection_model.pkl
├── uploads/
├── results/
└── Home.py
```

## Technical Requirements 💻

- Python 3.10 or later
- CUDA-capable GPU (recommended)
- CUDA Toolkit (for GPU acceleration)
- Minimum 8GB RAM

## Dependencies 📚

Major dependencies include:
- streamlit
- torch
- pyannote.audio
- whisper
- transformers
- librosa
- google-generativeai
- scikit-learn

See `requirements.txt` for complete list.

## Features in Detail 🔍

### Audio Analysis
- Speaker separation and identification
- High-quality speech-to-text conversion
- Real-time sentiment analysis
- Emotion detection from audio features

### Conversation Analysis
- Speaker interaction patterns
- Emotional tone mapping
- Sentiment progression
- Key topics identification

### Batch Processing
- Multiple file upload support
- Parallel processing capability
- Combined results in ZIP format
- Progress tracking for each file

## Output Format 📝

The analysis generates a structured summary including:
1. Main Topics
2. Conversation Dynamics
3. Speaker Analysis
   - Key Points
   - Predominant Emotions
   - Sentiments
   - Behavior Analysis
   - Tone
4. Overall Emotional Tone
5. Key Insights
6. Actionable Suggestions

## Troubleshooting 🔧

Common issues and solutions:

1. CUDA related warnings:
   - Ensure CUDA toolkit is installed
   - Update GPU drivers
   - Check CUDA compatibility with PyTorch version

2. Memory issues:
   - Reduce batch size
   - Process shorter audio segments
   - Close other GPU-intensive applications

3. Model loading errors:
   - Verify Hugging Face token
   - Check internet connection
   - Ensure model files are present

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 👏

- PyAnnote Audio for speaker diarization
- OpenAI Whisper for transcription
- Hugging Face for transformer models
- Google for Gemini API
- Streamlit for the web interface

## Author ✍️

[Harsh]
- GitHub: [@harsh-jha-a104911b5]
- Email: harshjha19.jsr@gmail.com

## Support 💪

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Open an issue on GitHub!
3. Contact the author 

## Implementation 

![Screenshot 2024-11-22 124136](https://github.com/user-attachments/assets/c1639477-a69c-44ea-b055-a50eb8242e84)
![Screenshot 2024-11-22 124314](https://github.com/user-attachments/assets/adb82aa7-821c-44c8-bc0c-33797737549a)
