"""
Smoke test CPU — vérifie le pipeline complet sans GPU ni données réelles.
Lance 10 steps d'entraînement sur des données synthétiques.

Usage :
    python tests/test_pipeline.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.config import get_paths, load_config
from src.dataset import DiffWaveDODaDataset, make_train_loader, make_val_loader
from src.diffusion import DiffusionSchedule
from src.losses import stft_loss
from src.trainer import Trainer

# ============================================================
# Helpers
# ============================================================
SAMPLE_RATE    = 22050
HOP_LENGTH     = 256
CROP_MEL_FRAMES = 62
AUDIO_LENGTH   = CROP_MEL_FRAMES * HOP_LENGTH   # 15 872
N_MELS         = 80


def _make_synthetic_data(tmp_dir: Path, n_examples: int = 10):
    """Crée de faux .wav, .npy et CSV dans tmp_dir."""
    wav_dir = tmp_dir / "wavs"
    mel_dir = tmp_dir / "mels"
    wav_dir.mkdir()
    mel_dir.mkdir()

    records = []
    for i in range(n_examples):
        uid  = f"{i:05d}"
        wav  = np.random.randn(AUDIO_LENGTH).astype(np.float32) * 0.1
        mel  = np.random.randn(N_MELS, CROP_MEL_FRAMES).astype(np.float32)

        wav_path = wav_dir / f"{uid}.wav"
        mel_path = mel_dir / f"{uid}.npy"
        sf.write(wav_path, wav, SAMPLE_RATE)
        np.save(mel_path, mel)

        records.append({
            "id":         uid,
            "speaker":    "M1",
            "text_latin": "test",
            "wav_path":   str(wav_path),
            "mel_path":   str(mel_path),
            "duration_s": round(AUDIO_LENGTH / SAMPLE_RATE, 3),
            "mel_frames": CROP_MEL_FRAMES,
        })

    df = pd.DataFrame(records)
    # 6 train, 2 val, 2 test
    train_df = df.iloc[:6].copy()
    val_df   = df.iloc[6:8].copy()
    test_df  = df.iloc[8:].copy()

    train_df.to_csv(tmp_dir / "train.csv", index=False)
    val_df.to_csv(  tmp_dir / "val.csv",   index=False)
    test_df.to_csv( tmp_dir / "test.csv",  index=False)

    summary = {
        "dataset":          "test",
        "exemples_traites": n_examples,
        "sample_rate":      SAMPLE_RATE,
        "n_mels":           N_MELS,
        "n_fft":            1024,
        "hop_length":       HOP_LENGTH,
        "splits":           {"train": 6, "val": 2, "test": 2},
    }
    with open(tmp_dir / "summary.json", "w") as f:
        json.dump(summary, f)

    return train_df, val_df, test_df, summary


# ============================================================
# Tests
# ============================================================
def test_dataset():
    print("Test 1 : DiffWaveDODaDataset.__getitem__")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_df, _, _, summary = _make_synthetic_data(tmp)

        dataset = DiffWaveDODaDataset(
            dataframe=train_df,
            wav_dir=tmp / "wavs",
            mel_dir=tmp / "mels",
            sample_rate=summary["sample_rate"],
            hop_length=summary["hop_length"],
            crop_mel_frames=CROP_MEL_FRAMES,
        )

        item = dataset[0]
        assert item["audio"].shape == torch.Size([AUDIO_LENGTH]), \
            f"Audio shape attendu [{AUDIO_LENGTH}], obtenu {item['audio'].shape}"
        assert item["mel"].shape == torch.Size([N_MELS, CROP_MEL_FRAMES]), \
            f"Mel shape attendu [{N_MELS}, {CROP_MEL_FRAMES}], obtenu {item['mel'].shape}"

        print(f"  audio : {item['audio'].shape} ✓")
        print(f"  mel   : {item['mel'].shape}   ✓")


def test_stft_loss():
    print("Test 2 : stft_loss")
    x = torch.randn(2, AUDIO_LENGTH)
    y = torch.randn(2, AUDIO_LENGTH)
    loss = stft_loss(x, y)
    assert loss.item() > 0, "STFT loss doit être positif"
    assert torch.isfinite(loss), "STFT loss doit être fini"
    print(f"  stft_loss : {loss.item():.6f} ✓")


def test_diffusion_schedule():
    print("Test 3 : DiffusionSchedule.add_noise + p_sample")
    try:
        from diffwave.params import params
        from diffwave.model  import DiffWave
    except ImportError:
        print("  diffwave non installé — test ignoré")
        return

    device   = "cpu"
    schedule = DiffusionSchedule(params.noise_schedule, device)
    model    = DiffWave(params).to(device)
    model.eval()

    audio    = torch.randn(1, AUDIO_LENGTH)
    mel      = torch.randn(1, N_MELS, CROP_MEL_FRAMES)
    t        = torch.zeros(1, dtype=torch.long)

    noisy, noise = schedule.add_noise(audio, t)
    assert noisy.shape == audio.shape
    assert torch.isfinite(noisy).all()
    print(f"  add_noise : {noisy.shape} ✓")

    with torch.no_grad():
        x_prev = schedule.p_sample(model, noisy, step=0, mel=mel)
    assert x_prev.shape == audio.shape
    assert torch.isfinite(x_prev).all()
    print(f"  p_sample  : {x_prev.shape} ✓")


def test_training_loop():
    print("Test 4 : boucle d'entraînement (10 steps, CPU)")
    try:
        from diffwave.params import params
        from diffwave.model  import DiffWave
    except ImportError:
        print("  diffwave non installé — test ignoré")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp     = Path(tmp)
        run_dir = tmp / "runs" / "smoke_test"

        train_df, val_df, _, summary = _make_synthetic_data(tmp)

        cfg = {
            "data_dir":          str(tmp),
            "runs_dir":          str(tmp / "runs"),
            "experiment_name":   "smoke_test",
            "batch_size":        2,
            "num_workers":       0,
            "learning_rate":     2e-4,
            "max_steps":         10,
            "log_every":         5,
            "valid_every":       5,
            "save_every":        5,
            "valid_max_batches": 2,
            "grad_clip":         1.0,
            "stft_weight":       0.1,
            "use_amp":           False,   # CPU : pas d'AMP
            "crop_mel_frames":   CROP_MEL_FRAMES,
        }

        paths = get_paths(cfg, mkdir=True)

        train_loader = make_train_loader(cfg, train_df, summary)
        val_loader   = make_val_loader(  cfg, val_df,   summary)

        device   = "gpu"
        model    = DiffWave(params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        schedule  = DiffusionSchedule(params.noise_schedule, device)

        trainer = Trainer(model, optimizer, schedule, cfg, paths, device, summary)
        trainer.train(train_loader, val_loader)

        # Vérifications
        best_path = paths["checkpoint_dir"] / "best.pt"
        assert best_path.exists(), "best.pt doit exister après l'entraînement"

        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        print(f"  best.pt sauvegardé au step {ckpt['step']} ✓")

        # Rechargement
        model2 = DiffWave(params)
        Trainer.load_checkpoint(str(best_path), model2, None, "cpu")
        print(f"  Rechargement du checkpoint ✓")


def test_generation():
    print("Test 5 : génération audio (CPU)")
    try:
        from diffwave.params import params
        from diffwave.model  import DiffWave
    except ImportError:
        print("  diffwave non installé — test ignoré")
        return

    device   = "cpu"
    model    = DiffWave(params).to(device)
    model.eval()
    schedule = DiffusionSchedule(params.noise_schedule, device)

    mel = torch.randn(1, N_MELS, CROP_MEL_FRAMES)
    with torch.no_grad():
        gen = schedule.generate(model, mel, AUDIO_LENGTH)

    assert gen.shape == torch.Size([1, AUDIO_LENGTH]), \
        f"Shape attendu [1, {AUDIO_LENGTH}], obtenu {gen.shape}"
    assert torch.isfinite(gen).all(), "Audio généré contient des NaN/Inf"
    print(f"  generate : {gen.shape} ✓")
    print(f"  Valeurs finies : ✓")


# ============================================================
# Runner
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("SMOKE TEST — DiffWave-Darija (CPU)")
    print("=" * 50)

    tests = [
        test_dataset,
        test_stft_loss,
        test_diffusion_schedule,
        test_training_loop,
        test_generation,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ECHEC : {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Résultat : {passed} réussi(s), {failed} échoué(s)")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)