"""
Script d'évaluation du modèle VoiceGAN
Calcule les métriques objectives (MCD, Similarité cosinus, etc.)
"""
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, DataConfig
from src.models.voicegan import VoiceGAN
from src.preprocessing.audio_processor import AudioProcessor
from src.training.dataset import create_dataloaders


def compute_mcd(mel1, mel2):
    """
    Calcule le Mel Cepstral Distortion entre deux melspectrograms

    Args:
        mel1, mel2: (n_mels, time) - Melspectrograms à comparer

    Returns:
        mcd: Scalaire - Distance MCD en dB
    """
    # Aligner les longueurs
    min_len = min(mel1.shape[1], mel2.shape[1])
    mel1 = mel1[:, :min_len]
    mel2 = mel2[:, :min_len]

    # Calcul MCD
    diff = mel1 - mel2
    mcd = np.sqrt(np.sum(diff ** 2, axis=0))
    mcd = np.mean(mcd)

    # Conversion en dB
    mcd_db = (10.0 / np.log(10)) * mcd

    return mcd_db


def compute_cosine_similarity(embedding1, embedding2):
    """
    Calcule la similarité cosinus entre deux embeddings

    Args:
        embedding1, embedding2: Vecteurs d'embeddings

    Returns:
        similarity: Scalaire entre -1 et 1
    """
    if torch.is_tensor(embedding1):
        embedding1 = embedding1.cpu().numpy()
    if torch.is_tensor(embedding2):
        embedding2 = embedding2.cpu().numpy()

    # Normaliser
    embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)

    # Similarité cosinus
    similarity = np.dot(embedding1, embedding2)

    return similarity


class Evaluator:
    """Classe pour l'évaluation du modèle"""

    def __init__(self, model, processor, device, config):
        self.model = model
        self.processor = processor
        self.device = device
        self.config = config

        self.model.eval()

    def evaluate_sample(self, source_mel, target_mel):
        """
        Évalue un échantillon

        Returns:
            dict avec les métriques
        """
        # Convertir en tenseurs
        source_mel_tensor = torch.from_numpy(source_mel).unsqueeze(0).float().to(self.device)
        target_mel_tensor = torch.from_numpy(target_mel).unsqueeze(0).float().to(self.device)

        # Conversion
        with torch.no_grad():
            converted_mel_tensor = self.model.convert_voice(source_mel_tensor, target_mel_tensor)

            # Extraire les embeddings
            target_style = self.model.get_style_representation(target_mel_tensor)
            converted_style = self.model.get_style_representation(converted_mel_tensor)

            source_content = self.model.get_content_representation(source_mel_tensor)
            converted_content = self.model.get_content_representation(converted_mel_tensor)

        converted_mel = converted_mel_tensor.squeeze().cpu().numpy()

        # Calculer les métriques
        metrics = {}

        # 1. MCD (Mel Cepstral Distortion)
        metrics['mcd_target'] = compute_mcd(converted_mel, target_mel)

        # 2. Similarité de style (cosine similarity)
        metrics['style_similarity'] = compute_cosine_similarity(
            target_style.squeeze(),
            converted_style.squeeze()
        )

        # 3. Préservation du contenu
        metrics['content_preservation'] = compute_cosine_similarity(
            source_content.mean(dim=1).squeeze(),
            converted_content.mean(dim=1).squeeze()
        )

        return metrics

    def evaluate_dataset(self, test_loader, max_samples=None):
        """
        Évalue sur tout le dataset de test

        Args:
            test_loader: DataLoader de test
            max_samples: Nombre maximum d'échantillons à évaluer (None = tous)

        Returns:
            dict avec les métriques moyennes et par échantillon
        """
        all_metrics = {
            'mcd_target': [],
            'style_similarity': [],
            'content_preservation': []
        }

        sample_count = 0

        for batch in tqdm(test_loader, desc="Évaluation"):
            if max_samples and sample_count >= max_samples:
                break

            source_mel = batch['source_mel'].squeeze().cpu().numpy()
            target_mel = batch['target_mel'].squeeze().cpu().numpy()

            # Évaluer
            metrics = self.evaluate_sample(source_mel, target_mel)

            # Accumuler
            for key, value in metrics.items():
                all_metrics[key].append(value)

            sample_count += 1

        # Calculer les moyennes
        avg_metrics = {
            key: np.mean(values) for key, values in all_metrics.items()
        }

        # Calculer les std
        std_metrics = {
            key: np.std(values) for key, values in all_metrics.items()
        }

        results = {
            'average': avg_metrics,
            'std': std_metrics,
            'all_samples': all_metrics,
            'num_samples': sample_count
        }

        return results

    def print_results(self, results):
        """Affiche les résultats de manière formatée"""
        print("\n" + "=" * 60)
        print("RÉSULTATS DE L'ÉVALUATION")
        print("=" * 60)
        print(f"Nombre d'échantillons: {results['num_samples']}")
        print("\nMétriques moyennes:")
        print("-" * 60)

        avg = results['average']
        std = results['std']

        print(f"MCD (Target):             {avg['mcd_target']:.4f} ± {std['mcd_target']:.4f} dB")
        print(f"Style Similarity:         {avg['style_similarity']:.4f} ± {std['style_similarity']:.4f}")
        print(f"Content Preservation:     {avg['content_preservation']:.4f} ± {std['content_preservation']:.4f}")

        print("\nInterprétation:")
        print("-" * 60)

        # MCD
        if avg['mcd_target'] < 6.0:
            print("✅ MCD: Excellent (< 6.0 dB)")
        elif avg['mcd_target'] < 8.0:
            print("✓ MCD: Bon (6.0-8.0 dB)")
        else:
            print("⚠ MCD: À améliorer (> 8.0 dB)")

        # Style Similarity
        if avg['style_similarity'] > 0.85:
            print("✅ Style Similarity: Excellent (> 0.85)")
        elif avg['style_similarity'] > 0.75:
            print("✓ Style Similarity: Bon (0.75-0.85)")
        else:
            print("⚠ Style Similarity: À améliorer (< 0.75)")

        # Content Preservation
        if avg['content_preservation'] > 0.90:
            print("✅ Content Preservation: Excellent (> 0.90)")
        elif avg['content_preservation'] > 0.80:
            print("✓ Content Preservation: Bon (0.80-0.90)")
        else:
            print("⚠ Content Preservation: À améliorer (< 0.80)")

        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Évaluer VoiceGAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Répertoire de sortie pour les résultats')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Nombre maximum d\'échantillons à évaluer')
    args = parser.parse_args()

    # Configuration
    config = ModelConfig()
    device = torch.device(config.DEVICE)

    print(f"Device: {device}")

    # Charger le modèle
    print("Chargement du modèle...")
    model = VoiceGAN(config).to(device)
    model.load_checkpoint(args.checkpoint)
    model.eval()
    print("✓ Modèle chargé")

    # Créer le processor
    processor = AudioProcessor(config)

    # Créer les dataloaders
    print("Chargement des données...")
    _, _, test_loader = create_dataloaders(config, DataConfig)
    print(f"✓ {len(test_loader)} batches de test")

    # Créer l'évaluateur
    evaluator = Evaluator(model, processor, device, config)

    # Évaluer
    print("\nÉvaluation en cours...")
    results = evaluator.evaluate_dataset(test_loader, args.max_samples)

    # Afficher les résultats
    evaluator.print_results(results)

    # Sauvegarder les résultats
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'evaluation_results.json'

    # Convertir les arrays numpy en listes pour JSON
    results_json = {
        'average': results['average'],
        'std': results['std'],
        'num_samples': results['num_samples']
    }

    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Résultats sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()