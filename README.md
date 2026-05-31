# VibeCheck AI: Speech Emotion Recognition (SER) using CNN

A deep learning-powered pipeline built with Python to recognize and classify human emotions from speech data. This project implements advanced audio signal processing using Mel-Spectrogram features combined with a 2D Convolutional Neural Network (CNN) architecture.

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.x
- **Deep Learning:** TensorFlow, Keras
- **Audio Processing:** Librosa, Soundfile
- **Data Science:** Feature Extraction, Signal Processing, NumPy, Matplotlib

## 🎙️ Audio Feature Extraction Pipeline
To transform raw audio waveforms (`.wav`) into a format that a Convolutional Neural Network can understand, the system performs the following digital signal processing steps:
1. **Mel-Spectrogram Generation:** Converts the audio signal from the time domain to the frequency domain (via STFT) and maps it onto the Mel scale.
2. **Delta & Delta-Delta Scaling:** Captures the dynamic changes and trajectories of the audio features over time.
