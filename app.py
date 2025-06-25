import streamlit as st
from keras.models import load_model
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

# Load model
model = load_model("emotion_model.h5")

# Ensure this matches your training label order
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear','disgust','surprised']

# Feature extraction function
def extract_features(file):
    try:
        y, sr = librosa.load(file, sr=None)
        if len(y) == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled.reshape(1, 40, 1)
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# UI
st.title("🎵 Emotion Classifier")
st.write("Upload a `.wav` file to detect emotion")

uploaded_file = st.file_uploader("Upload audio", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner('Extracting features...'):
        features = extract_features(uploaded_file)
    
    if features is None:
        st.error("Invalid or silent audio. Please upload a valid `.wav` file.")
    else:
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]
        predicted_emotion = emotion_labels[predicted_index]
        
        st.success(f"🎧 Predicted Emotion: **{predicted_emotion}**")
        st.write(f"🧠 Confidence: `{confidence:.2f}`")

        st.write("Prediction probabilities:")
        for i, label in enumerate(emotion_labels):
            st.write(f"- {label}: {prediction[0][i]:.4f}")
