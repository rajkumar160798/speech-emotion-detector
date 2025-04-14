import os
import librosa
import numpy as np
import pandas as pd

DATA_PATH = "data/RAVDESS"
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset():
    data = []
    for actor_dir in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                parts = filename.split("-")
                emotion_code = parts[2]
                emotion = EMOTION_LABELS.get(emotion_code)
                file_path = os.path.join(actor_path, filename)
                features = extract_features(file_path)
                if features is not None:
                    data.append([*features, emotion])
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = process_dataset()
    df.to_csv("data/emotion_features.csv", index=False)
    print("✅ MFCC features saved to data/emotion_features.csv")

    df = pd.read_csv("data/emotion_features.csv")
    print(df.head(5))  