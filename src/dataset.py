"""
Dataset PyTorch pour DiffWave × DODa.
Extrait du notebook 50k steps, cellules cell-10 à cell-12.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Mapping index → locuteur / genre (documentation DODa officielle)
# ============================================================
SPEAKER_RANGES = [
    (0,     999,   "F1", "female"),
    (1000,  1999,  "M3", "male"),
    (2000,  2730,  "F2", "female"),
    (2731,  2800,  "M1", "male"),
    (2801,  2999,  "M2", "male"),
    (3000,  3999,  "M2", "male"),
    (4000,  4999,  "M1", "male"),
    (5000,  5999,  "F3", "female"),
    (6000,  6999,  "M1", "male"),
    (7000,  7999,  "F4", "female"),
    (8000,  8999,  "F1", "female"),
    (9000,  9999,  "M2", "male"),
    (10000, 10999, "M1", "male"),
    (11000, 11999, "M1", "male"),
    (12000, 12350, "M2", "male"),
    (12351, 12742, "M1", "male"),
]


def get_speaker_info(idx: int):
    """Retourne (speaker_id, gender) pour un index DODa global."""
    for start, end, speaker, gender in SPEAKER_RANGES:
        if start <= idx <= end:
            return speaker, gender
    return "unknown", "unknown"


# ============================================================
# Dataset PyTorch
# ============================================================
class DiffWaveDODaDataset(Dataset):
    def __init__(self, dataframe, wav_dir, mel_dir,
                 sample_rate=22050, hop_length=256, crop_mel_frames=62):
        self.df = dataframe.reset_index(drop=True)
        self.wav_dir = Path(wav_dir)
        self.mel_dir = Path(mel_dir)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.crop_mel_frames = crop_mel_frames
        self.audio_length = crop_mel_frames * hop_length

    def __len__(self):
        return len(self.df)

    def _resolve_wav_path(self, row):
        """Essaie le chemin du CSV, sinon reconstruit depuis l'id (chemins Colab stale)."""
        wav_path = Path(row["wav_path"])
        if wav_path.exists():
            return wav_path
        file_id = str(row["id"]).zfill(5)
        return self.wav_dir / f"{file_id}.wav"

    def _resolve_mel_path(self, row):
        """Essaie le chemin du CSV, sinon reconstruit depuis l'id (chemins Colab stale)."""
        mel_path = Path(row["mel_path"])
        if mel_path.exists():
            return mel_path
        file_id = str(row["id"]).zfill(5)
        return self.mel_dir / f"{file_id}.npy"

    def _pad_audio(self, audio):
        if len(audio) < self.audio_length:
            pad_len = self.audio_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")
        elif len(audio) > self.audio_length:
            audio = audio[:self.audio_length]
        return audio

    def _pad_mel(self, mel):
        if mel.shape[1] < self.crop_mel_frames:
            pad_len = self.crop_mel_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_len)), mode="constant")
        elif mel.shape[1] > self.crop_mel_frames:
            mel = mel[:, :self.crop_mel_frames]
        return mel

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wav_path = self._resolve_wav_path(row)
        mel_path = self._resolve_mel_path(row)

        audio, _ = sf.read(wav_path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)
        mel = np.load(mel_path).astype(np.float32)

        # Crop synchronisé Mel / audio
        if mel.shape[1] > self.crop_mel_frames:
            max_start = mel.shape[1] - self.crop_mel_frames
            mel_start = np.random.randint(0, max_start + 1)
            mel_end = mel_start + self.crop_mel_frames

            audio_start = mel_start * self.hop_length
            audio_end = audio_start + self.audio_length

            mel = mel[:, mel_start:mel_end]
            audio = audio[audio_start:audio_end]

        audio = self._pad_audio(audio)
        mel = self._pad_mel(mel)

        return {
            "audio":      torch.tensor(audio, dtype=torch.float32),
            "mel":        torch.tensor(mel,   dtype=torch.float32),
            "id":         row["id"],
            "speaker":    row["speaker"],
            "text_latin": row.get("text_latin", ""),
        }


# ============================================================
# Chargement pleine durée (pour la génération)
# ============================================================
def load_full_sample(row, wav_dir, mel_dir, hop_length: int = 256) -> dict:
    """
    Charge un exemple complet (mel et audio NON croppés) pour la génération.

    Contrairement à DiffWaveDODaDataset.__getitem__() qui croppe tout à
    crop_mel_frames (62 frames ≈ 0,72s), cette fonction charge le mel
    entier pour produire un audio de durée réelle (1,5 – 5,5s selon DODa).

    audio_length est dérivé de mel.shape[1] pour garantir la synchronisation.
    """
    # Résolution des chemins (même fallback que _resolve_wav/mel_path)
    wav_path = Path(row["wav_path"])
    if not wav_path.exists():
        wav_path = Path(wav_dir) / f"{str(row['id']).zfill(5)}.wav"

    mel_path = Path(row["mel_path"])
    if not mel_path.exists():
        mel_path = Path(mel_dir) / f"{str(row['id']).zfill(5)}.npy"

    audio, _ = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    mel = np.load(mel_path).astype(np.float32)   # [80, F] complet

    # Aligner l'audio sur la longueur dérivée du mel
    audio_length = mel.shape[1] * hop_length
    if len(audio) < audio_length:
        audio = np.pad(audio, (0, audio_length - len(audio)), mode="constant")
    else:
        audio = audio[:audio_length]

    return {
        "audio":        torch.tensor(audio, dtype=torch.float32),  # [audio_length]
        "mel":          torch.tensor(mel,   dtype=torch.float32),  # [80, F]
        "id":           row["id"],
        "speaker":      row["speaker"],
        "text_latin":   row.get("text_latin", ""),
        "audio_length": audio_length,
        "mel_frames":   mel.shape[1],
    }


# ============================================================
# Factories DataLoader
# ============================================================
def make_train_loader(cfg: dict, train_df, summary: dict) -> DataLoader:
    dataset = DiffWaveDODaDataset(
        dataframe=train_df,
        wav_dir=Path(cfg["data_dir"]) / "wavs",
        mel_dir=Path(cfg["data_dir"]) / "mels",
        sample_rate=summary["sample_rate"],
        hop_length=summary["hop_length"],
        crop_mel_frames=cfg["crop_mel_frames"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )


def make_val_loader(cfg: dict, val_df, summary: dict) -> DataLoader:
    dataset = DiffWaveDODaDataset(
        dataframe=val_df,
        wav_dir=Path(cfg["data_dir"]) / "wavs",
        mel_dir=Path(cfg["data_dir"]) / "mels",
        sample_rate=summary["sample_rate"],
        hop_length=summary["hop_length"],
        crop_mel_frames=cfg["crop_mel_frames"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )


def make_test_loader(cfg: dict, test_df, summary: dict) -> DataLoader:
    dataset = DiffWaveDODaDataset(
        dataframe=test_df,
        wav_dir=Path(cfg["data_dir"]) / "wavs",
        mel_dir=Path(cfg["data_dir"]) / "mels",
        sample_rate=summary["sample_rate"],
        hop_length=summary["hop_length"],
        crop_mel_frames=cfg["crop_mel_frames"],
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
