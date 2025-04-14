# 🎙️ Real-Time Speech Emotion Detection System

This project detects emotions from human voice in real-time using deep learning. It leverages MFCC feature extraction from audio and a CNN-based model to classify emotions.

## Features
- Extracts MFCC features using `librosa`
- Trains emotion classifier using TensorFlow / PyTorch
- Evaluates with confusion matrix & ROC
- Real-time emotion detection with mic input
- Web app powered by Streamlit / FastAPI

## Dataset
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [CREMA-D](https://zenodo.org/record/3816440)

## Setup
```bash
pip install -r requirements.txt
