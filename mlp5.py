import streamlit as st
import numpy as np
import sounddevice as sd
import threading
import time
import joblib
import torch
from df import init_df, enhance
import librosa
from scipy.signal import butter, lfilter
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

# Load models and scalers
mlp_model = joblib.load("mlp_emotion_classifier_best_model2.joblib")
encoder = joblib.load("emotion_encoder.joblib")
scaler = joblib.load("feature_scaler.joblib")

# DeepFilterNet init
model, df_state, _ = init_df()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
SAMPLE_RATE = 16000
BLOCKSIZE = 16000
SILENCE_THRESHOLD = 0.015
DENOISE_BLEND = 1.0

# Shared buffers
audio_buffer = np.zeros(BLOCKSIZE)
cleaned_audio = np.zeros(BLOCKSIZE)
emotion_label = "Neutral"
confidence_score = 0.0
lock = threading.Lock()

# Functions
def bandpass_filter(data, low=300, high=3400, fs=SAMPLE_RATE):
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data)

def normalize(audio):
    return audio / (np.max(np.abs(audio)) + 1e-6)

def preprocess(audio):
    return normalize(bandpass_filter(audio))

def denoise(audio):
    tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        tensor_audio = tensor_audio.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        output = enhance(model, df_state, tensor_audio.cpu())
    return output.numpy().flatten()

def is_silence(audio):
    return np.sqrt(np.mean(audio ** 2)) < SILENCE_THRESHOLD

def extract_features_from_array(y, sr=16000):
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))
    features = np.array([])

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    features = np.hstack((features, mfccs))

    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, chroma))

    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    features = np.hstack((features, mel))

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    features = np.hstack((features, contrast))

    y_harm = librosa.effects.harmonic(y)
    tonnetz = np.mean(librosa.feature.tonnetz(y=y_harm, sr=sr).T, axis=0)
    features = np.hstack((features, tonnetz))

    return features

def audio_callback(indata, frames, time_info, status):
    global audio_buffer, cleaned_audio, emotion_label, confidence_score
    with lock:
        audio_buffer = indata[:, 0]
        processed = preprocess(audio_buffer)
        if is_silence(processed):
            denoised = np.zeros_like(processed)
        else:
            denoised = denoise(processed)

        cleaned_audio = DENOISE_BLEND * denoised + (1 - DENOISE_BLEND) * processed

        try:
            features = extract_features_from_array(cleaned_audio)
            scaled = scaler.transform([features])
            probs = mlp_model.predict_proba(scaled)[0]
            idx = np.argmax(probs)
            emotion_label = encoder.inverse_transform([idx])[0]
            confidence_score = probs[idx]
        except Exception as e:
            emotion_label = "Error"
            confidence_score = 0.0
            print("Emotion error:", e)

def start_audio_stream():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE):
        while True:
            time.sleep(0.1)

# Start audio stream once
if "audio_started" not in st.session_state:
    threading.Thread(target=start_audio_stream, daemon=True).start()
    st.session_state.audio_started = True

# === Streamlit layout ===
st.set_page_config(layout="wide")
st.title("🎧 Real-Time Emotion Detection")

st.markdown(f"### Emotion: **{emotion_label}**")
st.markdown(f"Confidence: `{confidence_score:.2f}`")

col1, col2 = st.columns(2)

# Plot waveform
with col1:
    st.markdown("#### Raw Audio")
    fig1, ax1 = plt.subplots()
    with lock:
        ax1.plot(audio_buffer)
        ax1.set_ylim([-1, 1])
    st.pyplot(fig1)

# Plot FFT
with col2:
    st.markdown("#### Cleaned FFT Spectrum")
    fig2, ax2 = plt.subplots()
    with lock:
        fft_vals = np.abs(rfft(cleaned_audio))
        freqs = rfftfreq(len(cleaned_audio), d=1 / SAMPLE_RATE)
        ax2.plot(freqs, fft_vals)
        ax2.set_xlim([0, 4000])
    st.pyplot(fig2)

st.markdown("Updating every 0.5s...")
time.sleep(0.5)
st.rerun()
