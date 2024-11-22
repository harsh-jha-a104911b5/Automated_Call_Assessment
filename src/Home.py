# Directory structure:
# app/
# ├── pages/
# │   └── 1_Batch_Analysis.py
# └── Home.py

# First, let's create the main Home.py (rename your current app.py to Home.py)
import streamlit as st
import os
from pathlib import Path
import time
from final import ConversationAnalyzer

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

# Initialize the analyzer in session state if it doesn't exist
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = ConversationAnalyzer()

def main():
    # Page title and description
    st.title("🎙️ Conversation Analysis")
    st.markdown("""
    Upload an audio file (.wav format) to analyze the conversation dynamics, emotions, and generate insights.
    
    ### Features:
    - Single file analysis
    - Detailed conversation insights
    - Emotion and sentiment analysis
    - Speaker diarization
    
    ### Available Pages:
    1. **Home (Current)**: Single file analysis
    2. **Batch Analysis**: Process multiple files simultaneously (up to 10 files)
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
                
                # Run the analysis
                status_text.text("Running conversation analysis...")
                progress_bar.progress(30)
                
                # Perform analysis
                summary = st.session_state.analyzer.analyze_audio(str(file_path))
                
                # Update progress - 70%
                status_text.text("Processing results...")
                progress_bar.progress(70)
                time.sleep(1)
                
                # Result file path
                result_file = RESULTS_DIR / "summary.txt"
                
                if result_file.exists():
                    # Update progress - 90%
                    status_text.text("Loading results...")
                    progress_bar.progress(90)
                    time.sleep(1)
                    
                    # Display results in an expander
                    with st.expander("Analysis Results", expanded=True):
                        st.markdown(summary)
                    
                    # Provide download button for results
                    st.download_button(
                        label="Download Results",
                        data=summary,
                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_analysis.txt",
                        mime="text/plain"
                    )
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                else:
                    st.error("Results file not found. There might have been an error during analysis.")
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                progress_bar.empty()
                status_text.empty()
            
            # Clean up uploaded file
            file_path.unlink(missing_ok=True)

if __name__ == "__main__":
    main()