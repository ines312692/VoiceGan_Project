from setuptools import setup, find_packages

setup(
    name="voicegan",
    version="1.0.0",
    description="VoiceGAN: Voice-to-Voice Conversion using GANs",
    author="Master Info - Deep Learning Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "streamlit>=1.28.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)