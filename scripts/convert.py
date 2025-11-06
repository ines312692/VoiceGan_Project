"""
Script de conversion simple pour VoiceGAN
Convertit un fichier audio source avec le style d'un fichier audio cible
"""
import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig
from src.models.voicegan import VoiceGAN
from src.preprocessing.audio_processor import AudioProcessor
import soundfile as sf


def convert_audio(source_path, target_path, output_path, checkpoint_path, config):
    """
    Convertit un audio source avec le style d'un audio cible

    Args:
        source_path: Chemin vers l'audio source (A)
        target_path: Chemin vers l'audio cible (B)
        output_path: Chemin de sortie pour l'audio converti
        checkpoint_path: Chemin vers le checkpoint du modèle
        config: Configuration du modèle
    """
    device = torch.device(config.DEVICE)

    # Charger le modèle
    print("Chargement du modèle...")
    model = VoiceGAN(config).to(device)
    model.load_checkpoint(checkpoint_path)
    model.eval()
    print("✓ Modèle chargé")

    # Créer le processor
    processor = AudioProcessor(config)

    # Charger les audios
    print(f"\nChargement de l'audio source: {source_path}")
    source_audio, _ = processor.load_audio(source_path, normalize=True, trim_silence=True)
    print(f"✓ Durée: {len(source_audio) / config.SAMPLE_RATE:.2f}s")

    print(f"\nChargement de l'audio cible: {target_path}")
    target_audio, _ = processor.load_audio(target_path, normalize=True, trim_silence=True)
    print(f"✓ Durée: {len(target_audio) / config.SAMPLE_RATE:.2f}s")

    # Générer les melspectrograms
    print("\nGénération des melspectrograms...")
    source_mel = processor.audio_to_mel(source_audio)
    target_mel = processor.audio_to_mel(target_audio)

    # Normaliser
    source_mel = processor.normalize_mel(source_mel, method='instance')
    target_mel = processor.normalize_mel(target_mel, method='instance')

    print(f"✓ Source mel shape: {source_mel.shape}")
    print(f"✓ Target mel shape: {target_mel.shape}")

    # Convertir en tenseurs
    source_mel_tensor = torch.from_numpy(source_mel).unsqueeze(0).float().to(device)
    target_mel_tensor = torch.from_numpy(target_mel).unsqueeze(0).float().to(device)

    # Conversion A→B
    print("\nConversion en cours...")
    with torch.no_grad():
        converted_mel_tensor = model.convert_voice(source_mel_tensor, target_mel_tensor)

    converted_mel = converted_mel_tensor.squeeze().cpu().numpy()
    print(f"✓ Converted mel shape: {converted_mel.shape}")

    # Dénormaliser
    converted_mel = processor.denormalize_mel(converted_mel, method='instance')

    # Générer l'audio final (Griffin-Lim)
    print("\nGénération de l'audio final...")
    converted_audio = processor.mel_to_audio(converted_mel, vocoder=None)

    # Sauvegarder
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output_path, converted_audio, config.SAMPLE_RATE)
    print(f"\n✅ Audio converti sauvegardé: {output_path}")
    print(f"   Durée: {len(converted_audio) / config.SAMPLE_RATE:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Convertir un audio source avec le style d\'un audio cible'
    )
    parser.add_argument('--source', type=str, required=True,
                        help='Chemin vers l\'audio source (A)')
    parser.add_argument('--target', type=str, required=True,
                        help='Chemin vers l\'audio cible (B)')
    parser.add_argument('--output', type=str, required=True,
                        help='Chemin de sortie pour l\'audio converti')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Chemin vers le checkpoint du modèle')
    args = parser.parse_args()

    # Configuration
    config = ModelConfig()

    print("\n" + "=" * 60)
    print("VoiceGAN-Transformation: Conversion A→B")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {config.DEVICE}")
    print("=" * 60)

    # Conversion
    try:
        convert_audio(
            args.source,
            args.target,
            args.output,
            args.checkpoint,
            config
        )
        print("\n✅ Conversion terminée avec succès!")

    except Exception as e:
        print(f"\n❌ Erreur lors de la conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()