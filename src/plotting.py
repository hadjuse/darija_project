"""
Fonctions de visualisation — sauvegarde les figures sans plt.show().
Extraites des cellules cell-29, cell-30, cell-45, cell-47 du notebook 50k.
"""
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def save_loss_curves(train_losses: list, val_losses: list,
                     figures_dir: Path, name: str) -> None:
    """
    Sauvegarde les courbes de loss d'entraînement et de validation.

    train_losses : liste de float (un par step)
    val_losses   : liste de dict {"step": int, "val_loss": float}
    """
    figures_dir = Path(figures_dir)

    # Courbe d'entraînement
    steps = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, train_losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Évolution de la loss d'entraînement — {name}")
    ax.grid(True)
    fig.savefig(figures_dir / "training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Courbe de validation
    if val_losses:
        val_steps = [d["step"] for d in val_losses]
        val_vals  = [d["val_loss"] for d in val_losses]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(val_steps, val_vals, marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("Validation MSE Loss")
        ax.set_title(f"Évolution de la loss de validation — {name}")
        ax.grid(True)
        fig.savefig(figures_dir / "validation_loss.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_waveform_comparison(orig_np: np.ndarray, gen_np: np.ndarray,
                              sample_id, figures_dir: Path, sample_rate: int) -> None:
    """Sauvegarde la comparaison waveform original vs généré."""
    figures_dir = Path(figures_dir)
    sample_id_str = str(sample_id).zfill(5)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(orig_np, label="Original", alpha=0.8)
    ax.plot(gen_np,  label="Généré",   alpha=0.7)
    ax.set_title(f"Waveform original vs généré — ID {sample_id_str}")
    ax.set_xlabel("Échantillons")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    fig.savefig(
        figures_dir / f"waveform_original_vs_generated_{sample_id_str}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def save_mel_comparison(cond_mel: np.ndarray, gen_audio_np: np.ndarray,
                         sample_id, figures_dir: Path,
                         sample_rate: int, hop_length: int,
                         n_fft: int, n_mels: int, f_max: float,
                         clip_val: float = -11.5) -> None:
    """Sauvegarde la comparaison Mel conditionnel vs Mel de l'audio généré."""
    figures_dir = Path(figures_dir)
    sample_id_str = str(sample_id).zfill(5)

    # Calcul du Mel de l'audio généré
    gen_mel = librosa.feature.melspectrogram(
        y=gen_audio_np,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        power=1.0,
    )
    gen_mel_log = np.log(np.clip(gen_mel, 1e-5, None))
    gen_mel_log = np.clip(gen_mel_log, clip_val, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    librosa.display.specshow(
        cond_mel, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=axes[0],
    )
    axes[0].set_title("Mel conditionnel")

    librosa.display.specshow(
        gen_mel_log, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=axes[1],
    )
    axes[1].set_title("Mel de l'audio généré")

    fig.suptitle(f"Comparaison Mel — ID {sample_id_str}")
    plt.tight_layout()
    fig.savefig(
        figures_dir / f"mel_condition_vs_generated_{sample_id_str}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
