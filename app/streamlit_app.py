import streamlit as st
import torch
import torchaudio
from pathlib import Path
import sys
import tempfile
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import Config
from src.models.voicegan import VoiceGAN
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.mel_spectrogram import MelSpectrogramProcessor

# Page config
st.set_page_config(
    page_title="VoiceGAN - Voice Conversion A→B",
    page_icon="",
    layout="wide"
)

# Title
st.title(" VoiceGAN: Voice Conversion System")
st.markdown("Transform voice A into voice B while preserving content")

# Sidebar for settings
with st.sidebar:
    st.header(" Settings")

    checkpoint_path = st.text_input(
        "Model Checkpoint Path",
        value="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )

    config_path = st.text_input(
        "Config Path",
        value="config/config.yaml",
        help="Path to configuration file"
    )

    device = st.selectbox(
        "Device",
        ["cuda", "cpu"],
        index=0 if torch.cuda.is_available() else 1
    )

    st.markdown("---")
    st.markdown("### About")
    st.info(
        """
        This system converts the voice of speaker A to speaker B:
        - **Content** from source audio (what is said)
        - **Style** from target audio (how it sounds)
        - Uses CNN + Transformer + GAN architecture
        """
    )

@st.cache_resource
def load_model(checkpoint_path, config_path, device):
    """Load model (cached)"""
    try:
        config = Config(config_path)

        model = VoiceGAN(
            n_mels=config.audio.n_mels,
            content_channels=config.content_encoder.channels,
            content_kernel_sizes=config.content_encoder.kernel_sizes,
            content_strides=config.content_encoder.strides,
            transformer_dim=config.content_encoder.transformer_dim,
            num_heads=config.content_encoder.num_heads,
            num_transformer_layers=config.content_encoder.num_layers,
            style_channels=config.style_encoder.channels,
            style_kernel_sizes=config.style_encoder.kernel_sizes,
            style_strides=config.style_encoder.strides,
            style_dim=config.style_encoder.style_dim,
            generator_channels=config.generator.channels,
            generator_kernel_sizes=config.generator.kernel_sizes,
            upsample_rates=config.generator.upsample_rates
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Initialize processors
        audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            segment_length=config.audio.segment_length
        )

        mel_processor = MelSpectrogramProcessor(
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            n_mels=config.audio.n_mels,
            fmin=config.audio.fmin,
            fmax=config.audio.fmax
        )

        return model, audio_processor, mel_processor, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def convert_voice(model, audio_processor, mel_processor, source_audio, target_audio, device):
    """Perform voice conversion"""
    try:
        # Convert to mel spectrograms
        source_mel = mel_processor.wav_to_mel(source_audio).unsqueeze(0).to(device)
        target_mel = mel_processor.wav_to_mel(target_audio).unsqueeze(0).to(device)

        # Convert
        with torch.no_grad():
            converted_mel = model.convert(source_mel, target_mel)

        # Convert back to audio
        converted_audio = mel_processor.mel_to_wav(converted_mel.squeeze(0).cpu())

        return converted_audio
    except Exception as e:
        st.error(f"Error during conversion: {e}")
        return None

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header(" Source Audio (A)")
    source_file = st.file_uploader(
        "Upload source audio",
        type=['wav', 'mp3', 'flac'],
        key='source',
        help="The voice you want to convert FROM"
    )

    if source_file:
        st.audio(source_file, format='audio/wav')

with col2:
    st.header(" Target Reference (B)")
    target_file = st.file_uploader(
        "Upload target reference audio",
        type=['wav', 'mp3', 'flac'],
        key='target',
        help="The voice you want to convert TO"
    )

    if target_file:
        st.audio(target_file, format='audio/wav')

# Convert button
if st.button(" Convert Voice A → B", type="primary", use_container_width=True):
    if source_file is None or target_file is None:
        st.error("Please upload both source and target audio files!")
    else:
        with st.spinner("Loading model..."):
            model, audio_processor, mel_processor, config = load_model(
                checkpoint_path, config_path, device
            )

        if model is None:
            st.error("Failed to load model. Please check paths and configuration.")
        else:
            with st.spinner("Converting voice..."):
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_source:
                    tmp_source.write(source_file.read())
                    source_path = tmp_source.name

                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_target:
                    tmp_target.write(target_file.read())
                    target_path = tmp_target.name

                # Load audio
                source_audio = audio_processor.load_audio(source_path)
                target_audio = audio_processor.load_audio(target_path)

                # Convert
                converted_audio = convert_voice(
                    model, audio_processor, mel_processor,
                    source_audio, target_audio, device
                )

                if converted_audio is not None:
                    # Save converted audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_output:
                        output_path = tmp_output.name

                    audio_processor.save_audio(converted_audio, output_path)

                    # Display result
                    st.success(" Conversion completed!")
                    st.header("🎵 Converted Audio (A→B)")

                    # Play audio
                    st.audio(output_path, format='audio/wav')

                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label=" Download Converted Audio",
                            data=f.read(),
                            file_name="converted_voice.wav",
                            mime="audio/wav"
                        )

                    # Visualizations
                    st.subheader(" Mel-Spectrograms Comparison")

                    fig_col1, fig_col2, fig_col3 = st.columns(3)

                    with fig_col1:
                        st.markdown("**Source (A)**")
                        source_mel = mel_processor.wav_to_mel(source_audio)
                        st.image(
                            source_mel.numpy(),
                            caption="Source Mel-Spectrogram",
                            use_container_width=True,
                            clamp=True
                        )

                    with fig_col2:
                        st.markdown("**Target (B)**")
                        target_mel = mel_processor.wav_to_mel(target_audio)
                        st.image(
                            target_mel.numpy(),
                            caption="Target Mel-Spectrogram",
                            use_container_width=True,
                            clamp=True
                        )

                    with fig_col3:
                        st.markdown("**Converted (A→B)**")
                        converted_mel = mel_processor.wav_to_mel(converted_audio)
                        st.image(
                            converted_mel.numpy(),
                            caption="Converted Mel-Spectrogram",
                            use_container_width=True,
                            clamp=True
                        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>VoiceGAN Project - Master Info Deep Learning 2025-2026</p>
        <p>Université de Tunis</p>
    </div>
    """,
    unsafe_allow_html=True
)