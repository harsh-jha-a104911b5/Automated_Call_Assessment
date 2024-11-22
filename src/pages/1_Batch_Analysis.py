# pages/1_Batch_Analysis.py
import streamlit as st
import os
from pathlib import Path
import time
from final import ConversationAnalyzer
import zipfile
import io

# Set page configuration
st.set_page_config(
    page_title="Batch Conversation Analysis",
    page_icon="📚",
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

def process_audio_file(file_path, progress_bar, status_text, progress_range):
    """Process a single audio file and update progress within the given range"""
    start_progress, end_progress = progress_range
    progress_span = end_progress - start_progress
    
    try:
        # Update progress - Starting file
        status_text.text(f"Processing: {file_path.name}")
        progress_bar.progress(start_progress + (0.1 * progress_span))
        
        # Perform analysis
        summary = st.session_state.analyzer.analyze_audio(str(file_path))
        
        # Update progress - Completing file
        progress_bar.progress(end_progress)
        
        return {
            'filename': file_path.name,
            'summary': summary,
            'status': 'success'
        }
    
    except Exception as e:
        return {
            'filename': file_path.name,
            'summary': f"Error processing file: {str(e)}",
            'status': 'error'
        }

def main():
    st.title("📚 Batch Conversation Analysis")
    st.markdown("""
    Upload multiple audio files (up to 10) for batch processing. Each file will be analyzed for conversation dynamics, 
    emotions, and insights.
    """)
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose WAV files (up to 10 files)",
        type=['wav'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Check number of files
        if len(uploaded_files) > 10:
            st.error("Please upload a maximum of 10 files.")
            return
        
        # Display selected files
        st.write(f"Selected {len(uploaded_files)} files:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
        
        # Create analyze button
        if st.button("Analyze All Files"):
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create containers for results
            results_container = st.container()
            
            try:
                results = []
                
                # Calculate progress segments for each file
                progress_per_file = 1 / len(uploaded_files)
                
                # Process each file
                for index, uploaded_file in enumerate(uploaded_files):
                    # Calculate progress range for this file
                    start_progress = index * progress_per_file
                    end_progress = (index + 1) * progress_per_file
                    
                    # Save the uploaded file
                    file_path = UPLOAD_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file
                    result = process_audio_file(
                        file_path,
                        progress_bar,
                        status_text,
                        (start_progress, end_progress)
                    )
                    
                    results.append(result)
                    
                    # Clean up uploaded file
                    file_path.unlink(missing_ok=True)
                
                # Create ZIP file containing all results
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for result in results:
                        filename = f"{result['filename'].rsplit('.', 1)[0]}_analysis.txt"
                        zip_file.writestr(filename, result['summary'])
                
                # Display results
                with results_container:
                    st.success("Batch processing complete!")
                    
                    # Display individual results in expanders
                    for result in results:
                        with st.expander(f"Results for {result['filename']}", expanded=False):
                            if result['status'] == 'success':
                                st.markdown(result['summary'])
                            else:
                                st.error(result['summary'])
                    
                    # Provide download button for all results
                    st.download_button(
                        label="Download All Results (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="batch_analysis_results.zip",
                        mime="application/zip"
                    )
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("All files processed!")
                
            except Exception as e:
                st.error(f"An error occurred during batch processing: {str(e)}")
                progress_bar.empty()
                status_text.empty()

if __name__ == "__main__":
    main()