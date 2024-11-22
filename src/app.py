import streamlit as st
import os
import subprocess
import time
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Conversation Analysis",
    page_icon="🎙️",
    layout="wide"
)

# Create necessary directories if they don't exist
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    # Page title and description
    st.title("🎙️ Conversation Analysis")
    st.markdown("""
    Upload an audio file (.wav format) to analyze the conversation dynamics, emotions, and generate insights.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        # Create an analyze button
        if st.button("Analyze Conversation"):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress bar - 10%
                status_text.text("Initializing analysis...")
                progress_bar.progress(10)
                time.sleep(1)
                
                # Run the analysis script
                status_text.text("Running conversation analysis...")
                progress_bar.progress(30)
                
                # Run final.py with the uploaded file path as input
                process = subprocess.Popen(
                    ['python', 'final.py'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Provide the file path to the script
                stdout, stderr = process.communicate(input=str(file_path))
                
                if process.returncode != 0:
                    raise Exception(f"Analysis failed: {stderr}")
                
                # Update progress - 70%
                status_text.text("Processing results...")
                progress_bar.progress(70)
                time.sleep(1)
                
                # Get the results file path (same name as input but .txt extension)
                result_file = RESULTS_DIR / "summary.txt"
                
                if result_file.exists():
                    # Update progress - 90%
                    status_text.text("Loading results...")
                    progress_bar.progress(90)
                    time.sleep(1)
                    
                    # Read and display results
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_text = f.read()
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    # Display results in an expander
                    with st.expander("Analysis Results", expanded=True):
                        st.markdown(result_text)
                    
                    # Provide download button for results
                    st.download_button(
                        label="Download Results",
                        data=result_text,
                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_analysis.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Results file not found. There might have been an error during analysis.")
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                progress_bar.empty()
                status_text.empty()
            
            # Clean up uploaded file
            file_path.unlink(missing_ok=True)
    
    # Add some information about the supported analysis
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool analyzes conversations from audio files and provides:
        - Speaker diarization
        - Transcription
        - Sentiment analysis
        - Emotion detection
        - Conversation dynamics analysis
        
        **Supported Format:**
        - WAV audio files
        """)

if __name__ == "__main__":
    main()