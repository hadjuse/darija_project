"""
Génération audio depuis un checkpoint DiffWave entraîné.
Peut être exécuté indépendamment de l'entraînement.

Usage :
    python scripts/generate.py \
        --checkpoint runs/diffwave_doda_scratch/checkpoints/best.pt \
        [--config configs/default.yaml] \
        [--data_dir data_preprocessed] \
        [--output_dir runs/diffwave_doda_scratch/generated_audios] \
        [--num_examples 5]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import soundfile as sf
import torch
from diffwave.model import DiffWave
from diffwave.params import params

from src.config import get_paths, load_config
from src.dataset import load_full_sample
from src.diffusion import DiffusionSchedule
from src.plotting import save_mel_comparison, save_waveform_comparison


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Génération audio DiffWave × DODa")
    parser.add_argument("--checkpoint",   required=True,
                        help="Chemin vers le checkpoint .pt")
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--data_dir",     default=None)
    parser.add_argument("--output_dir",   default=None,
                        help="Répertoire de sortie (défaut : generated_audios/ du run)")
    parser.add_argument("--num_examples", type=int, default=None)
    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    # Config
    ckpt_path = Path(args.checkpoint)
    raw_ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Priorité : default.yaml < checkpoint cfg (contient les vrais hyperparamètres d'entraînement)
    base_cfg  = load_config("configs/default.yaml")
    saved_cfg = raw_ckpt.get("cfg", {})
    cfg       = {**base_cfg, **saved_cfg}

    # Surcharges CLI
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.num_examples is not None:
        cfg["num_examples"] = args.num_examples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Checkpoint : {ckpt_path}")

    # Données
    data_dir = Path(cfg["data_dir"])
    test_df  = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "summary.json", "r") as f:
        summary = json.load(f)

    # Répertoire de sortie
    output_dir = Path(args.output_dir) if args.output_dir else (
        ckpt_path.parent.parent / "generated_audios"
    )
    figures_dir = ckpt_path.parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Modèle
    model = DiffWave(params).to(device)
    model.load_state_dict(raw_ckpt["model_state_dict"])
    model.eval()
    print(f"Modèle chargé — step {raw_ckpt.get('step', '?')}")

    # Schedule
    schedule = DiffusionSchedule(params.noise_schedule, device)

    # Génération — mel complet (pleine durée, non croppé)
    num_examples = cfg.get("num_examples", 5)
    generated_results = []

    print(f"\nGénération de {num_examples} exemples (pleine durée)...")
    for i in range(min(num_examples, len(test_df))):
        row    = test_df.iloc[i]
        sample = load_full_sample(
            row,
            data_dir / "wavs",
            data_dir / "mels",
            hop_length=summary["hop_length"],
        )

        mel_i      = sample["mel"].unsqueeze(0).to(device)
        orig_audio = sample["audio"].numpy()
        sample_id  = sample["id"]
        sample_id_str = str(sample_id).zfill(5)

        audio_length = sample["audio_length"]   # durée réelle de l'utterance
        duration_s   = audio_length / summary["sample_rate"]
        print(f"  [{i+1}/{num_examples}] ID {sample_id_str} — {sample['mel_frames']} frames mel — {duration_s:.2f}s")

        gen_audio = schedule.generate(model, mel_i, audio_length)
        gen_np    = gen_audio[0].cpu().numpy()

        # Sauvegarde WAV
        out_path = output_dir / f"generated_{sample_id_str}.wav"
        sf.write(out_path, gen_np, summary["sample_rate"])

        # Comparaisons visuelles
        save_waveform_comparison(
            orig_np=orig_audio, gen_np=gen_np,
            sample_id=sample_id, figures_dir=figures_dir,
            sample_rate=summary["sample_rate"],
        )
        save_mel_comparison(
            cond_mel=sample["mel"].numpy(), gen_audio_np=gen_np,
            sample_id=sample_id, figures_dir=figures_dir,
            sample_rate=summary["sample_rate"],
            hop_length=summary["hop_length"],
            n_fft=summary["n_fft"],
            n_mels=summary["n_mels"],
            f_max=cfg.get("f_max", 8000.0),
            clip_val=cfg.get("clip_val", -11.5),
        )

        generated_results.append({
            "id":                  sample_id_str,
            "speaker":             sample["speaker"],
            "text_latin":          sample["text_latin"],
            "duration_s":          round(duration_s, 3),
            "mel_frames":          sample["mel_frames"],
            "generated_audio_path":str(out_path),
        })

    # CSV résultats
    results_df = pd.DataFrame(generated_results)
    csv_path   = output_dir.parent / "results_csv" / "generated_audio_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    print(f"\nGénération terminée. Fichiers dans : {output_dir}")
    print(f"Résultats CSV : {csv_path}")


if __name__ == "__main__":
    main()
