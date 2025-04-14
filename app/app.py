import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import joblib

st.title("🎤 Real-Time Speech Emotion Detection")

model = tf.keras.models.load_model("src/emotion_model.h5")
label_encoder = joblib.load("src/label_encoder.pkl")

SAMPLE_RATE = 22050
DURATION = 3

def record_audio():
    st.info(f"Recording for {DURATION} seconds...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio.flatten()

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

if st.button("Start Recording"):
    audio = record_audio()
    features = extract_features(audio)
    norm_features = tf.keras.utils.normalize([features])
    prediction = model.predict(norm_features)
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    def emotion_style(emotion):
        emoji_map = {
            'happy': '😄', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
            'disgust': '🤢', 'surprised': '😲', 'calm': '😌', 'neutral': '😐'
        }
        return f"{emoji_map.get(emotion, '')} **Predicted Emotion: `{emotion}`**"

    st.success(emotion_style(emotion))
