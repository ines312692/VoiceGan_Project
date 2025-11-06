"""
Application Streamlit pour la démonstration de VoiceGAN-Transformation
Interface interactive pour la conversion de voix A→B
"""
import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import plotly.graph_objects as go
from pathlib import Path
import sys
import io

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from src.models.voicegan import VoiceGAN
from src.preprocessing.audio_processor import AudioProcessor

# Configuration de la page
st.set_page_config(
    page_title="VoiceGAN-Transformation",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path):
    """Charge le modèle VoiceGAN"""
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VoiceGAN(config).to(device)

    try:
        model.load_checkpoint(checkpoint_path)
        model.eval()
        return model, config, device
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None, None, None


def plot_spectrogram(mel, title="Melspectrogram", sr=22050, hop_length=256):
    """Crée un plot interactif du melspectrogram"""

    # Convertir en numpy si nécessaire
    if torch.is_tensor(mel):
        mel = mel.cpu().numpy()

    # Calcul des axes temporels
    times = librosa.frames_to_time(np.arange(mel.shape[1]),
                                   sr=sr, hop_length=hop_length)

    # Créer le plot
    fig = go.Figure(data=go.Heatmap(
        z=mel,
        x=times,
        y=np.arange(mel.shape[0]),
        colorscale='Viridis',
        colorbar=dict(title="dB")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Temps (s)",
        yaxis_title="Mel Frequency Bin",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def convert_voice(model, processor, device, source_audio, target_audio):
    """
    Convertit la voix source avec le style de la voix cible

    Returns:
        converted_audio, source_mel, target_mel, converted_mel
    """
    # Générer les melspectrograms
    source_mel = processor.audio_to_mel(source_audio)
    target_mel = processor.audio_to_mel(target_audio)

    # Normaliser
    source_mel = processor.normalize_mel(source_mel)
    target_mel = processor.normalize_mel(target_mel)

    # Convertir en tenseurs
    source_mel_tensor = torch.from_numpy(source_mel).unsqueeze(0).float().to(device)
    target_mel_tensor = torch.from_numpy(target_mel).unsqueeze(0).float().to(device)

    # Conversion A→B
    with torch.no_grad():
        converted_mel_tensor = model.convert_voice(source_mel_tensor, target_mel_tensor)

    # Convertir en numpy
    converted_mel = converted_mel_tensor.squeeze().cpu().numpy()

    # Dénormaliser
    converted_mel = processor.denormalize_mel(converted_mel)

    # Générer l'audio (Griffin-Lim)
    converted_audio = processor.mel_to_audio(converted_mel)

    return converted_audio, source_mel, target_mel, converted_mel


def main():
    # Titre
    st.title("🎤 VoiceGAN-Transformation")
    st.markdown("### Conversion de Voix A→B avec Deep Learning")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Sélection du checkpoint
        checkpoint_path = st.text_input(
            "Chemin du checkpoint",
            "checkpoints/best_model.pth",
            help="Chemin vers le fichier de checkpoint du modèle"
        )

        # Charger le modèle
        if st.button("🔄 Charger le modèle"):
            with st.spinner("Chargement du modèle..."):
                model, config, device = load_model(checkpoint_path)
                if model is not None:
                    st.session_state['model'] = model
                    st.session_state['config'] = config
                    st.session_state['device'] = device
                    st.session_state['processor'] = AudioProcessor(config)
                    st.success("✅ Modèle chargé avec succès!")

        st.markdown("---")

        # Informations
        st.header("📊 Informations")
        if 'model' in st.session_state:
            st.write(f"**Device:** {st.session_state['device']}")
            st.write(f"**Sample Rate:** {st.session_state['config'].SAMPLE_RATE} Hz")
            st.write(f"**N_mels:** {st.session_state['config'].N_MELS}")
        else:
            st.info("Veuillez charger le modèle")

        st.markdown("---")
        st.markdown("**📚 Documentation**")
        st.markdown("- [GitHub Repository](#)")
        st.markdown("- [Paper](#)")

    # Corps principal
    if 'model' not in st.session_state:
        st.warning("⚠️ Veuillez charger le modèle dans le panneau latéral")
        st.info("""
        ### Comment utiliser l'application:
        1. Chargez le modèle dans le panneau latéral
        2. Uploadez un fichier audio source (Voix A)
        3. Uploadez un fichier audio cible (Voix B)
        4. Cliquez sur 'Convertir' pour générer la voix transformée
        """)
        return

    # Récupérer les objets de session
    model = st.session_state['model']
    config = st.session_state['config']
    device = st.session_state['device']
    processor = st.session_state['processor']

    # Deux colonnes pour les uploads
    col1, col2 = st.columns(2)

    with col1:
        st.header("🎙️ Voix Source (A)")
        source_file = st.file_uploader(
            "Upload audio source",
            type=['wav', 'mp3', 'flac'],
            key="source",
            help="La voix que vous voulez transformer"
        )

        if source_file is not None:
            # Lire l'audio
            source_audio, sr = librosa.load(io.BytesIO(source_file.read()),
                                            sr=config.SAMPLE_RATE)

            # Afficher l'audio
            st.audio(source_file, format='audio/wav')

            # Informations
            duration = len(source_audio) / sr
            st.write(f"**Durée:** {duration:.2f}s")
            st.write(f"**Échantillons:** {len(source_audio)}")

    with col2:
        st.header("🎯 Voix Cible (B)")
        target_file = st.file_uploader(
            "Upload audio cible",
            type=['wav', 'mp3', 'flac'],
            key="target",
            help="Le style vocal que vous voulez appliquer"
        )

        if target_file is not None:
            # Lire l'audio
            target_audio, sr = librosa.load(io.BytesIO(target_file.read()),
                                            sr=config.SAMPLE_RATE)

            # Afficher l'audio
            st.audio(target_file, format='audio/wav')

            # Informations
            duration = len(target_audio) / sr
            st.write(f"**Durée:** {duration:.2f}s")
            st.write(f"**Échantillons:** {len(target_audio)}")

    # Bouton de conversion
    st.markdown("---")

    if source_file is not None and target_file is not None:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

        with col_btn2:
            if st.button("🚀 Convertir la voix A→B", use_container_width=True):
                with st.spinner("Conversion en cours... Veuillez patienter."):
                    try:
                        # Conversion
                        converted_audio, source_mel, target_mel, converted_mel = convert_voice(
                            model, processor, device, source_audio, target_audio
                        )

                        # Sauvegarder dans la session
                        st.session_state['converted_audio'] = converted_audio
                        st.session_state['source_mel'] = source_mel
                        st.session_state['target_mel'] = target_mel
                        st.session_state['converted_mel'] = converted_mel

                        st.success("✅ Conversion réussie!")

                    except Exception as e:
                        st.error(f"❌ Erreur lors de la conversion: {e}")

    # Afficher les résultats
    if 'converted_audio' in st.session_state:
        st.markdown("---")
        st.header("🎵 Résultat de la conversion")

        # Audio converti
        converted_audio = st.session_state['converted_audio']

        # Créer un buffer pour l'audio
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, converted_audio, config.SAMPLE_RATE, format='WAV')
        audio_buffer.seek(0)

        # Afficher l'audio
        st.audio(audio_buffer, format='audio/wav')

        # Bouton de téléchargement
        st.download_button(
            label="⬇️ Télécharger l'audio converti",
            data=audio_buffer,
            file_name="converted_voice.wav",
            mime="audio/wav"
        )

        # Visualisations
        st.markdown("---")
        st.header("📊 Visualisations")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Source", "Cible", "Converti", "Comparaison"
        ])

        with tab1:
            fig = plot_spectrogram(
                st.session_state['source_mel'],
                "Melspectrogram Source (A)",
                config.SAMPLE_RATE,
                config.HOP_LENGTH
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = plot_spectrogram(
                st.session_state['target_mel'],
                "Melspectrogram Cible (B)",
                config.SAMPLE_RATE,
                config.HOP_LENGTH
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = plot_spectrogram(
                st.session_state['converted_mel'],
                "Melspectrogram Converti (A→B)",
                config.SAMPLE_RATE,
                config.HOP_LENGTH
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Comparaison côte à côte
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Source")
                fig1 = plot_spectrogram(
                    st.session_state['source_mel'],
                    "Source",
                    config.SAMPLE_RATE,
                    config.HOP_LENGTH
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.subheader("Cible")
                fig2 = plot_spectrogram(
                    st.session_state['target_mel'],
                    "Cible",
                    config.SAMPLE_RATE,
                    config.HOP_LENGTH
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col3:
                st.subheader("Converti")
                fig3 = plot_spectrogram(
                    st.session_state['converted_mel'],
                    "Converti",
                    config.SAMPLE_RATE,
                    config.HOP_LENGTH
                )
                st.plotly_chart(fig3, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>VoiceGAN-Transformation | Master Info - Deep Learning | 2025-2026</p>
        <p>Université de Tunis | Dr. Mehrez Boulares</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()