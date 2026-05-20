"""
Pipeline de prétraitement DODa → wavs + mels + CSV splits.
Extrait du notebook 01_preprocessing_doda.ipynb.

Usage :
    python scripts/preprocess.py [--config configs/default.yaml] \
                                  [--data_dir data_preprocessed] \
                                  [--seed 42]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.config import load_config
from src.dataset import SPEAKER_RANGES, get_speaker_info


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Prétraitement DODa pour DiffWave")
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--data_dir", default=None,
                        help="Répertoire de sortie (écrase data_dir dans la config)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--hf_token", default=None,
                        help="Token HuggingFace (sinon env HF_TOKEN ou prompt)")
    return parser.parse_args()


# ============================================================
# Pipeline audio
# ============================================================
def make_mel_transform(cfg: dict):
    return T.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        n_mels=cfg["n_mels"],
        f_min=cfg.get("f_min", 0.0),
        f_max=cfg["f_max"],
        power=cfg.get("power", 1.0),
    )


def preprocess_audio(audio_dict: dict, cfg: dict, mel_transform) -> tuple:
    """
    Entrée  : dict HuggingFace {"array": ndarray, "sampling_rate": int}
    Sortie  : (wav Tensor[1,T], mel Tensor[80,F])
    Étapes  : mono → resample → peak norm → Mel → log-compression
    """
    wav    = torch.tensor(audio_dict["array"], dtype=torch.float32)
    src_sr = audio_dict["sampling_rate"]
    target_sr = cfg["sample_rate"]
    log_offset = cfg.get("log_offset", 1e-5)
    clip_val   = cfg["clip_val"]

    # Mono
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Rééchantillonnage
    if src_sr != target_sr:
        wav = T.Resample(orig_freq=src_sr, new_freq=target_sr)(wav)

    # Normalisation peak
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.95

    # Mel-spectrogramme
    mel = mel_transform(wav).squeeze(0)   # [80, F]

    # Log-compression
    mel = torch.log(mel.clamp(min=log_offset)).clamp(min=clip_val)

    return wav, mel


def process_corpus(dataset_filtered, global_indices: list, cfg: dict,
                   wav_dir: Path, mel_dir: Path, mel_transform) -> pd.DataFrame:
    """Applique le pipeline sur tout le corpus et sauvegarde les fichiers."""
    manifest = []
    sample_rate = cfg["sample_rate"]

    for local_i, global_i in enumerate(tqdm(global_indices, desc="Preprocessing")):
        item   = dataset_filtered[local_i]
        spk, _ = get_speaker_info(global_i)
        uid    = f"{global_i:05d}"

        wav_t, mel_t = preprocess_audio(item["audio"], cfg, mel_transform)

        wav_path = wav_dir / f"{uid}.wav"
        mel_path = mel_dir / f"{uid}.npy"

        sf.write(wav_path, wav_t.squeeze().numpy(), sample_rate, subtype="PCM_16")
        np.save(mel_path, mel_t.numpy().astype(np.float32))

        manifest.append({
            "id":         uid,
            "speaker":    spk,
            "text_latin": item.get("darija_Latn", ""),
            "wav_path":   str(wav_path),
            "mel_path":   str(mel_path),
            "duration_s": round(wav_t.shape[-1] / sample_rate, 3),
            "mel_frames": mel_t.shape[-1],
        })

    return pd.DataFrame(manifest)


def split_dataset(df: pd.DataFrame, test_size=0.10, val_size=0.10,
                  seed=42) -> tuple:
    """Split stratifié par locuteur : 80/10/10."""
    train_val, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["speaker"]
    )
    train_df, val_df = train_test_split(
        train_val, test_size=val_size, random_state=seed,
        stratify=train_val["speaker"]
    )
    return train_df, val_df, test_df


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    cfg = load_config(
        args.config,
        cli_overrides={"data_dir": args.data_dir} if args.data_dir else None,
    )

    data_dir = Path(cfg["data_dir"])
    wav_dir  = data_dir / "wavs"
    mel_dir  = data_dir / "mels"
    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    print(f"Répertoire de sortie : {data_dir.resolve()}")

    # Authentification HuggingFace (optionnelle pour les datasets publics)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    else:
        print("Pas de token HuggingFace — chargement en accès public (dataset DODa est public).")

    # Chargement DODa
    print("\nChargement du dataset DODa...")
    dataset = load_dataset("atlasia/DODa-audio-dataset", split="train")
    print(f"DODa chargé : {len(dataset)} exemples")

    # Filtrage voix masculines
    male_indices    = [i for i in range(len(dataset)) if get_speaker_info(i)[1] == "male"]
    dataset_male    = dataset.select(male_indices)
    print(f"Voix masculines retenues : {len(male_indices)} (M1, M2, M3)")

    # Prétraitement
    mel_transform = make_mel_transform(cfg)
    print("\nPrétraitement audio en cours...")
    df_manifest = process_corpus(dataset_male, male_indices, cfg, wav_dir, mel_dir, mel_transform)
    print(f"Exemples traités : {len(df_manifest)}")
    print(f"Durée totale : {df_manifest['duration_s'].sum() / 3600:.2f} h")

    # Splits
    train_df, val_df, test_df = split_dataset(df_manifest, seed=args.seed)
    print(f"\nSplits — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(  data_dir / "val.csv",   index=False)
    test_df.to_csv( data_dir / "test.csv",  index=False)

    # Résumé JSON
    summary = {
        "dataset":          "atlasia/DODa-audio-dataset",
        "filtre":           "voix masculines (M1, M2, M3)",
        "exemples_traites": len(df_manifest),
        "duree_totale_h":   round(df_manifest["duration_s"].sum() / 3600, 3),
        "locuteurs":        sorted(df_manifest["speaker"].unique().tolist()),
        "sample_rate":      cfg["sample_rate"],
        "n_mels":           cfg["n_mels"],
        "n_fft":            cfg["n_fft"],
        "hop_length":       cfg["hop_length"],
        "splits":           {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    with open(data_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nPrétraitement terminé. Fichiers dans : {data_dir.resolve()}")
    print("Prochaine étape : python scripts/train.py --config configs/train_scratch.yaml")


if __name__ == "__main__":
    main()
