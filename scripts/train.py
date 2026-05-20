"""
Entraînement DiffWave sur DODa (from-scratch ou fine-tuning).
Reproduit l'expérience du notebook Diffwave_training_generation50k_steps.ipynb.

Usage (from-scratch 50k) :
    python scripts/train.py --config configs/train_scratch.yaml

Usage (fine-tuning depuis LJSpeech) :
    python scripts/train.py --config configs/train_finetune.yaml \
                            --pretrained_path /chemin/vers/ljspeech.pt

Usage (reprendre un entraînement) :
    python scripts/train.py --config configs/train_scratch.yaml \
                            --resume runs/diffwave_doda_scratch/checkpoints/last.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from diffwave.model import DiffWave
from diffwave.params import params

from src.config import get_paths, load_config
from src.dataset import make_train_loader, make_val_loader
from src.diffusion import DiffusionSchedule
from src.plotting import save_loss_curves
from src.trainer import Trainer


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement DiffWave × DODa")
    parser.add_argument("--config",          default=None,
                        help="Fichier YAML d'override (appliqué par-dessus configs/default.yaml)")
    parser.add_argument("--default_config",  default="configs/default.yaml",
                        help="Fichier YAML de base (défaut : configs/default.yaml)")
    parser.add_argument("--data_dir",        default=None)
    parser.add_argument("--runs_dir",        default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--max_steps",       type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--learning_rate",   type=float, default=None)
    parser.add_argument("--log_every",       type=int,   default=None)
    parser.add_argument("--valid_every",     type=int,   default=None)
    parser.add_argument("--save_every",      type=int,   default=None)
    parser.add_argument("--pretrained_path", default=None,
                        help="Checkpoint LJSpeech pour fine-tuning (None = from scratch)")
    parser.add_argument("--resume",          default=None,
                        help="Checkpoint pour reprendre un entraînement")
    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k not in ("config", "default_config", "resume") and v is not None
    }

    # default.yaml toujours chargé en base ; --config est l'override expérience
    cfg = load_config(args.default_config, args.config, cli_overrides)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Expérience : {cfg['experiment_name']}")
    print(f"Max steps : {cfg['max_steps']}")

    # Chemins
    paths = get_paths(cfg, mkdir=True)

    # Données
    train_df = pd.read_csv(paths["train_csv"])
    val_df   = pd.read_csv(paths["val_csv"])
    with open(paths["summary_json"], "r") as f:
        summary = json.load(f)

    print(f"\nTrain : {len(train_df)} | Val : {len(val_df)}")

    train_loader = make_train_loader(cfg, train_df, summary)
    val_loader   = make_val_loader(cfg, val_df, summary)

    print(f"Batches train : {len(train_loader)} | Batches val : {len(val_loader)}")

    # Modèle
    model = DiffWave(params).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModèle DiffWave — {num_params:,} paramètres entraînables")

    # Fine-tuning : chargement de poids pré-entraînés
    if cfg.get("pretrained_path"):
        print(f"Chargement des poids pré-entraînés : {cfg['pretrained_path']}")
        ckpt = torch.load(cfg["pretrained_path"], map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Clés manquantes : {len(missing)} | Clés inattendues : {len(unexpected)}")

    # Optimizer + schedule de diffusion
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    schedule  = DiffusionSchedule(params.noise_schedule, device)

    # Reprise d'entraînement
    start_step    = 0
    train_losses  = []
    val_losses    = []

    if args.resume:
        print(f"\nReprise depuis : {args.resume}")
        ckpt = Trainer.load_checkpoint(args.resume, model, optimizer, device)
        start_step   = ckpt.get("step", 0)
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses",   [])
        print(f"  Reprise au step {start_step}")

    # Entraînement
    trainer = Trainer(model, optimizer, schedule, cfg, paths, device, summary)
    trainer.global_step  = start_step
    trainer.train_losses = train_losses
    trainer.val_losses   = val_losses

    print(f"\nDébut de l'entraînement...")
    trainer.train(train_loader, val_loader)

    # Sauvegarde des loss CSV
    loss_df = pd.DataFrame({
        "step":       list(range(1, len(trainer.train_losses) + 1)),
        "train_loss": trainer.train_losses,
    })
    loss_df.to_csv(paths["results_csv_dir"] / "training_losses.csv", index=False)

    if trainer.val_losses:
        val_df_out = pd.DataFrame(trainer.val_losses)
        val_df_out.to_csv(paths["results_csv_dir"] / "validation_losses.csv", index=False)

    print(f"\nRésultats sauvegardés dans : {paths['run_dir']}")


if __name__ == "__main__":
    main()
