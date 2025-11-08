#  Guide de Démarrage Rapide - VoiceGAN

## Installation (5 minutes)

```bash
# 1. Cloner le repository
git clone <repository-url>
cd VoiceGan_Project

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
pip install -e .
```

## Télécharger des Données (Option rapide)

### Option 1: Dataset VCTK (Recommandé)
```bash
# Télécharger VCTK
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip

# Extraire
unzip VCTK-Corpus-0.92.zip -d raw_data/

# Préparer les données
python scripts/prepare_data.py \
    --input_dir raw_data/VCTK-Corpus/wav48_silence_trimmed \
    --output_dir data/
```

### Option 2: Vos propres enregistrements
```bash
# Structure requise:
# raw_data/
#   speaker_A/
#     audio1.wav
#     audio2.wav
#   speaker_B/
#     audio1.wav
#     audio2.wav

python scripts/prepare_data.py \
    --input_dir raw_data/ \
    --output_dir data/
```

## Entraînement Rapide (Configuration minimale)

### 1. Éditer la configuration (optionnel)
```bash
nano config/config.yaml
```

Points clés à ajuster:
- `batch_size`: Réduire à 4-8 si mémoire limitée
- `num_epochs`: 50-100 pour tests rapides
- `learning_rate_g/d`: Laisser par défaut

### 2. Lancer l'entraînement
```bash
# Mode standard
python scripts/train.py --data_dir data/ --device cuda

# Mode CPU (plus lent)
python scripts/train.py --data_dir data/ --device cpu

# Avec monitoring TensorBoard
tensorboard --logdir logs/
# Ouvrir http://localhost:6006
```

### 3. Surveiller l'entraînement
```bash
# Dans un autre terminal
tensorboard --logdir logs/
```

Métriques à surveiller:
- `g_recon` (↓): Qualité de reconstruction
- `g_adv` (→ stable): Équilibre GAN
- `d_total` (→ ~0.5): Discriminateur équilibré

## Conversion Vocale (2 minutes)

### Via ligne de commande
```bash
python scripts/convert.py \
    --source exemples/source_audio.wav \
    --target exemples/target_reference.wav \
    --output converted_output.wav \
    --checkpoint checkpoints/best_model.pt
```

### Via interface Web (Recommandé)
```bash
streamlit run app/streamlit_app.py
```

1. Ouvrir http://localhost:8501
2. Uploader audio source (A)
3. Uploader référence cible (B)
4. Cliquer "Convert"
5. Télécharger le résultat

## Évaluation

```bash
python scripts/evaluate.py \
    --test_dir data/test \
    --checkpoint checkpoints/best_model.pt \
    --output_dir outputs/evaluation \
    --save_audio
```

Résultats dans `outputs/evaluation/`:
- `evaluation_results.json`: Métriques numériques
- `audio_samples/`: Exemples audio

## Exemples de Résultats Attendus

### Après 10 epochs
- MCD: ~10-12 dB (acceptable)
- Similarité: 0.6-0.7
- Audio: Légèrement robotique

### Après 50 epochs
- MCD: ~7-9 dB (bon)
- Similarité: 0.75-0.85
- Audio: Naturel avec quelques artefacts

### Après 100+ epochs
- MCD: ~5-7 dB (excellent)
- Similarité: 0.85-0.95
- Audio: Très naturel

## Résolution de Problèmes Courants

###  CUDA out of memory
```yaml
# config/config.yaml
training:
  batch_size: 4  # Réduire de 16 à 4
  
audio:
  segment_length: 8192  # Réduire de 16384
```

###  Training instable (pertes divergent)
```yaml
training:
  learning_rate_g: 0.0001  # Réduire de 0.0002
  learning_rate_d: 0.00005  # Réduire de 0.0001
  discriminator_start_epoch: 10  # Augmenter de 5
```

###  Style pas transféré
```yaml
training:
  lambda_identity: 10.0  # Augmenter de 5.0
  lambda_content: 0.5  # Réduire de 1.0
```

###  Contenu pas préservé
```yaml
training:
  lambda_content: 2.0  # Augmenter de 1.0
  lambda_reconstruction: 15.0  # Augmenter de 10.0
```

## Pipeline de Développement Complet

### Jour 1: Setup & Exploration
```bash
# 1. Installation
pip install -r requirements.txt

# 2. Explorer données
jupyter notebook notebooks/01_data_exploration.ipynb

# 3. Test preprocessing
jupyter notebook notebooks/02_preprocessing.ipynb
```

### Jour 2-3: Entraînement Initial
```bash
# Quick test (10 epochs)
python scripts/train.py --data_dir data/ --num_epochs 10

# Vérifier outputs
python scripts/evaluate.py --test_dir data/test --checkpoint checkpoints/checkpoint_epoch_10.pt
```

### Jour 4-7: Entraînement Complet
```bash
# Full training
python scripts/train.py --data_dir data/ --num_epochs 100

# Monitor avec TensorBoard
tensorboard --logdir logs/
```

### Jour 8: Évaluation & Fine-tuning
```bash
# Évaluation complète
python scripts/evaluate.py \
    --test_dir data/test \
    --checkpoint checkpoints/best_model.pt \
    --save_audio

# Ajuster hyperparamètres si nécessaire
# Reprendre entraînement
python scripts/train.py --resume checkpoints/checkpoint_epoch_100.pt
```

### Jour 9: Démo & Documentation
```bash
# Préparer démo
streamlit run app/streamlit_app.py

# Générer exemples pour rapport
python scripts/convert.py --source ... --target ... --output demo_samples/
```

## Checklist Projet

- [ ] Installation complète
- [ ] Données préparées (train/val/test)
- [ ] Config ajustée pour votre machine
- [ ] Entraînement lancé (>50 epochs)
- [ ] TensorBoard configuré
- [ ] Évaluation effectuée (MCD, similarité)
- [ ] Interface Streamlit testée
- [ ] Exemples audio sauvegardés
- [ ] Rapport rédigé
- [ ] Schéma pipeline créé

## Ressources Utiles

### Documentation
- `README.md`: Documentation complète
- `docs/`: Documentation technique
- `notebooks/`: Tutoriels interactifs

### Support
- Issues GitHub: Pour bugs
- Documentation PyTorch: https://pytorch.org/docs
- Articles de référence: Voir README

## Prochaines Étapes

1.  Complétez ce quickstart
2.  Lisez le README complet
3.  Explorez les notebooks
4.  Ajustez la config pour vos besoins
5.  Lancez un entraînement complet
6.  Analysez les résultats
7.  Documentez vos expériences

Bon développement ! 🎤