"""
Launch the Streamlit web application.

Usage:
    streamlit run app/main.py
    OR
    python run_app.py
"""

import subprocess
import sys


def main():
    print("Launching Brain Tumor Agent web app...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app/main.py",
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()
