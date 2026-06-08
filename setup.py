"""
Package setup script.

Optional: allows installing the project as a Python package
so imports work from anywhere.

Usage:
    pip install -e .    # Install in development mode
"""

from setuptools import setup, find_packages

setup(
    name="brain-tumor-agent",
    version="0.1.0",
    description="Reinforcement Learning Based Brain Tumor Agent using Vision Transformers",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "stable-baselines3>=2.1.0",
        "gymnasium>=0.29.0",
        "streamlit>=1.28.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
    ],
)
