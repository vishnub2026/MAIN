# 🎤 குரல் உணர்ச்சி கண்டுபிடிப்பு - 2 முறைகள் (Mic + File Upload)
# Save as: app_dual_mode.py
# Run with: streamlit run app_dual_mode.py

import os
import librosa
import numpy as np
import streamlit as st
import sounddevice as sd
import wavio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Feature Extraction
# ---------------------------
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# ---------------------------
# 2. English → Tamil Emotion Mapping
# ---------------------------
emotion_map = {
    "happy": "மகிழ்ச்சி",
    "sad": "சோகம்",
    "angry": "கோபம்",
    "neutral": "சாதாரணம்",
    "calm": "அமைதி",
    "fear": "பயம்",
    "disgust": "வெறுப்பு",
    "surprise": "ஆச்சரியம்"
}

# ---------------------------
# 3. Load/Train Model
# ---------------------------
@st.cache_resource
def train_demo_model(dataset_path="dataset/"):
    data, labels = [], []
    for emotion_dir in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion_dir)
        if os.path.isdir(emotion_path):
            for file in os.listdir(emotion_path):
                if file.endswith(".wav"):
                    try:
                        features = extract_features(os.path.join(emotion_path, file))
                        data.append(features)
                        labels.append(emotion_dir.lower())
                    except:
                        pass

    X, y = np.array(data), np.array(labels)
    if len(X) == 0:
        st.error("❌ Dataset-ல் audio files இல்லை. தயவு செய்து .wav files சேர்க்கவும்.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, list(set(y))

model, emotions = train_demo_model()

# ---------------------------
# 4. Streamlit UI
# ---------------------------
st.title("🎤 Identify the Emotions from the audio.")
st.write("👉 There are  2  Modes is here:   1) 🎙️ Mic Input   2) 📂 .wav File Upload")

mode = st.radio("Select your Mode:", ("🎙️ Mic Input", "📂 File Upload"))

# ---------------------------
# Mode 1: Mic Input
# ---------------------------
if mode == "🎙️ Mic Input":
    st.write("🎙️ 3 விநாடிகள் பேசுங்கள் (Record button-ஐ அழுத்தவும்)")

    if st.button("🎙️ Record"):
        duration = 3  # seconds
        fs = 44100  # sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wavio.write("mic_input.wav", recording, fs, sampwidth=2)
        st.success("✅ குரல் பதிவு செய்யப்பட்டது!")

        st.audio("mic_input.wav", format="audio/wav")

        features = extract_features("mic_input.wav")
        prediction = model.predict([features])[0]
        tamil_emotion = emotion_map.get(prediction, prediction)
        st.success(f"🔮 கண்டறியப்பட்ட உணர்ச்சி: **{tamil_emotion}**")

# ---------------------------
# Mode 2: File Upload
# ---------------------------
elif mode == "📂 File Upload":
    uploaded_file = st.file_uploader(" `.wav` Upload 1 File", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        features = extract_features("temp.wav")
        prediction = model.predict([features])[0]
        tamil_emotion = emotion_map.get(prediction, prediction)
        st.success(f"🔮 கண்டறியப்பட்ட உணர்ச்சி: **{tamil_emotion}**")
