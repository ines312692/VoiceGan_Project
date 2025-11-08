#!/usr/bin/env python
"""Data preparation script"""

import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with raw audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for organized data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of validation data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def organize_data(input_dir: Path, output_dir: Path, train_ratio: float, val_ratio: float, seed: int):
    """Organize audio files by speaker and split into train/val/test"""

    random.seed(seed)

    # Find all audio files
    print("Scanning for audio files...")
    audio_extensions = ['.wav', '.flac', '.mp3']

    # Assume directory structure: input_dir/speaker_id/*.wav
    speakers = {}

    for speaker_dir in input_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            audio_files = []

            for ext in audio_extensions:
                audio_files.extend(list(speaker_dir.glob(f'*{ext}')))

            if audio_files:
                speakers[speaker_id] = audio_files

    print(f"Found {len(speakers)} speakers")

    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Split and copy files
    for speaker_id, files in tqdm(speakers.items(), desc="Processing speakers"):
        # Shuffle files
        random.shuffle(files)

        # Split
        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # Copy files
        for split_name, split_files, split_dir in [
            ('train', train_files, train_dir),
            ('val', val_files, val_dir),
            ('test', test_files, test_dir)
        ]:
            if split_files:
                speaker_out_dir = split_dir / speaker_id
                speaker_out_dir.mkdir(exist_ok=True)

                for file_path in split_files:
                    dst = speaker_out_dir / file_path.name
                    shutil.copy2(file_path, dst)

        print(f"  {speaker_id}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    print(f"\nData organized in: {output_dir}")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    print(f"  Test: {test_dir}")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print("Error: train_ratio + val_ratio must be <= 1.0")
        return

    print(f"Data split: Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}, Test={test_ratio:.1%}")

    organize_data(input_dir, output_dir, args.train_ratio, args.val_ratio, args.seed)

    print("\nData preparation completed!")


if __name__ == '__main__':
    main()