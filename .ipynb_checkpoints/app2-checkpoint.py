import torch
import torchaudio
import streamlit as st
from transformers import AutoModelForAudioClassification, AutoProcessor
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Configure page
st.set_page_config(
    page_title="🎧 Emotion Detection from Audio",
    page_icon="🎧",
    layout="wide"
)

# Title and instructions
st.title("🎧 Emotion Detection from Audio")
st.markdown("""
**Record audio or upload a `.wav` file to detect emotions and visualize spectrograms.**

This app uses a fine-tuned Audio Spectrogram Transformer (AST) model to classify emotions from audio files.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
show_waveform = st.sidebar.checkbox("Show Waveform", value=True)
show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
use_plotly = st.sidebar.checkbox("Use Interactive Plots", value=True)
recording_duration = st.sidebar.slider("Max Recording Duration (seconds)", 5, 60, 10)

# Load model and processor
@st.cache_resource
def load_model():
    try:
        model_path = "./ast-finetuned-model"
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForAudioClassification.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval().to(device)
        st.sidebar.success(f"✅ Model loaded on {device.upper()}")
        return processor, model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load the model
try:
    processor, model, device = load_model()
    
    # Display model info in sidebar
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.write(f"**Device:** {device.upper()}")
    if hasattr(model.config, 'id2label'):
        emotions = list(model.config.id2label.values())
        st.sidebar.write(f"**Emotions:** {', '.join(emotions)}")
    
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def process_audio_file(uploaded_file):
    """Process uploaded audio file and return waveform and sample rate."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load audio
        waveform, sr = torchaudio.load(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        return waveform, sr
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None, None

def process_audio(waveform, sr, audio_name="Audio"):
    """Process audio and display results."""
    try:
        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Display audio info
        duration = waveform.shape[1] / sr
        st.info(f"📊 **{audio_name}** - Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
        
        # Show waveform
        if show_waveform:
            st.markdown("### 🌊 Waveform")
            
            if use_plotly:
                # Interactive waveform plot
                time_axis = np.linspace(0, duration, waveform.shape[1])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_axis,
                    y=waveform.squeeze().numpy(),
                    mode='lines',
                    name='Waveform',
                    line=dict(color='blue', width=1)
                ))
                fig.update_layout(
                    title="Audio Waveform",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Amplitude",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Static matplotlib plot
                fig, ax = plt.subplots(figsize=(12, 4))
                time_axis = np.linspace(0, duration, waveform.shape[1])
                ax.plot(time_axis, waveform.squeeze().numpy(), color='blue', linewidth=0.5)
                ax.set_title("Audio Waveform")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Show spectrogram
        if show_spectrogram:
            st.markdown("### 🔊 Spectrogram")
            
            # Generate spectrogram
            spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
            spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
            
            if use_plotly:
                # Interactive spectrogram
                fig = go.Figure(data=go.Heatmap(
                    z=spectrogram_db.squeeze().numpy(),
                    colorscale='Viridis',
                    colorbar=dict(title="dB")
                ))
                fig.update_layout(
                    title="Mel Spectrogram (dB)",
                    xaxis_title="Time",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Static spectrogram
                fig, ax = plt.subplots(figsize=(12, 6))
                im = ax.imshow(spectrogram_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
                ax.set_title("Mel Spectrogram (dB)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Frequency")
                plt.colorbar(im, ax=ax, label='dB')
                st.pyplot(fig)
        
        # Predict emotion
        st.markdown("### 🎭 Emotion Prediction")
        
        with st.spinner("Analyzing emotion..."):
            # Convert waveform to numpy
            waveform_np = waveform.squeeze().numpy()
            
            # Preprocess for model
            inputs = processor(waveform_np, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            pred_id = logits.argmax(-1).item()
            confidence = probabilities[0][pred_id].item()
            label = model.config.id2label[pred_id]
            
            # Display main result
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"🎭 **Detected Emotion: {label}**")
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Show all confidence scores
            if show_confidence:
                st.markdown("### 📊 All Emotion Scores")
                
                # Prepare data for visualization
                emotions = list(model.config.id2label.values())
                scores = probabilities.squeeze().cpu().numpy()
                
                if use_plotly:
                    # Interactive bar chart
                    fig = go.Figure(data=[
                        go.Bar(x=emotions, y=scores, 
                               marker_color=['#e74c3c' if i == pred_id else '#3498db' for i in range(len(emotions))])
                    ])
                    fig.update_layout(
                        title="Emotion Confidence Scores",
                        xaxis_title="Emotions",
                        yaxis_title="Confidence",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Static bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#e74c3c' if i == pred_id else '#3498db' for i in range(len(emotions))]
                    bars = ax.bar(emotions, scores, color=colors)
                    ax.set_title("Emotion Confidence Scores")
                    ax.set_xlabel("Emotions")
                    ax.set_ylabel("Confidence")
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Show detailed scores in a table
                scores_df = {
                    'Emotion': emotions,
                    'Confidence': [f"{score:.4f}" for score in scores],
                    'Percentage': [f"{score:.2%}" for score in scores]
                }
                st.dataframe(scores_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.exception(e)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload File"])

with tab1:
    st.markdown("### 🎙️ Record Your Voice")
    st.markdown("Click the record button below to start recording. The app will automatically analyze your audio when you stop recording.")
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#34495e",
        icon_name="microphone-lines",
        icon_size="2x",
        pause_threshold=2.0,
        sample_rate=16000
    )
    
    if audio_bytes:
        st.success("✅ Audio recorded successfully!")
        st.audio(audio_bytes, format="audio/wav")
        
        # Process recorded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            waveform, sr = torchaudio.load(tmp_path)
            os.unlink(tmp_path)  # Clean up
            
            # Process the audio
            process_audio(waveform, sr, "Recorded Audio")
            
        except Exception as e:
            st.error(f"Error processing recorded audio: {str(e)}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

with tab2:
    st.markdown("### 📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Upload an audio file", 
        type=["wav", "mp3", "m4a", "flac"],
        help="Supported formats: WAV, MP3, M4A, FLAC. WAV files at 16kHz work best."
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        # Process uploaded file
        waveform, sr = process_audio_file(uploaded_file)
        if waveform is not None and sr is not None:
            process_audio(waveform, sr, uploaded_file.name)