import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import joblib
import tensorflow as tf
import time

# Load model and label encoder
model = tf.keras.models.load_model("src/emotion_model.h5")
label_encoder = joblib.load("src/label_encoder.pkl")  # We'll save this now

SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 40

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print(f"🎙️ Recording {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

def extract_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def predict_emotion():
    audio = record_audio()
    mfcc_features = extract_mfcc(audio)
    mfcc_features = tf.keras.utils.normalize([mfcc_features])
    prediction = model.predict(mfcc_features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    print(f"🧠 Predicted Emotion: {predicted_label}")

if __name__ == "__main__":
    while True:
        predict_emotion()
        again = input("Try again? (y/n): ")
        if again.lower() != 'y':
            break
    print("👋 Goodbye!")
    
