import os
import subprocess

def main():
    port = int(os.environ.get('PORT', 4000))
    command = f"streamlit run Home.py --server.port={port} --server.address=0.0.0.0"
    
    print(f"Starting Streamlit app on port {port}")
    subprocess.run(command.split())

if __name__ == "__main__":
    main()