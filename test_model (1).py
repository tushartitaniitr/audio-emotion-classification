from keras.models import load_model
import numpy as np
import librosa

# Load the trained model
model = load_model("emotion_model.h5")

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, 40, 1)

# Path to test file (you can upload one or use a test example)
file_path = "test_audio.wav"  # Change this to your file name
features = extract_features(file_path)

# Predict
prediction = model.predict(features)
emotion_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear','disgust','surprised']
predicted_class = emotion_classes[np.argmax(pred)]
print("Predicted Emotion:", predicted_class)
