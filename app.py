import torch
import torchaudio
import streamlit as st
from transformers import AutoModelForAudioClassification, AutoProcessor
import matplotlib.pyplot as plt

# Title and instructions
st.title("🎧 Emotion Detection from Audio")
st.markdown("Upload a `.wav` file (16kHz recommended) to detect the emotion and visualize its spectrogram.")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

# Load model and processor
@st.cache_resource
def load_model():
    model_path = "./ast-finetuned-model"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

processor, model = load_model()

# Run prediction if file is uploaded
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Load and resample audio
    waveform, sr = torchaudio.load(uploaded_file)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono with shape (1, N)

    # Display spectrogram
    st.markdown("### 🔊 Spectrogram")
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Mel Spectrogram (dB)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.imshow(spectrogram_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    st.pyplot(fig)

    # Convert waveform to numpy
    waveform_np = waveform.squeeze().numpy()

    # Preprocess for model
    inputs = processor(waveform_np, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(-1).item()
    label = model.config.id2label[pred_id]

    # Display result
    st.success(f"🎙️ Detected Emotion: **{label}**")
