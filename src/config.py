"""
Gestion de la configuration : chargement YAML + surcharges argparse.
"""
from pathlib import Path
import yaml


def load_config(default_yaml: str, override_yaml: str = None, cli_overrides: dict = None) -> dict:
    """
    Fusionne :
      1. default_yaml  — base (toujours chargé)
      2. override_yaml — fichier d'expérience (optionnel)
      3. cli_overrides — dict de surcharges CLI (valeurs non-None uniquement)

    Retourne un dict complet de configuration.
    """
    with open(default_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    if override_yaml is not None:
        with open(override_yaml, "r") as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)

    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None:
                cfg[key] = val

    return cfg


def get_paths(cfg: dict, mkdir: bool = False) -> dict:
    """
    Construit tous les chemins du projet à partir de la config.
    Si mkdir=True, crée les répertoires de sortie.
    """
    data_dir = Path(cfg["data_dir"])
    run_dir = Path(cfg["runs_dir"]) / cfg["experiment_name"]

    paths = {
        "data_dir":       data_dir,
        "wav_dir":        data_dir / "wavs",
        "mel_dir":        data_dir / "mels",
        "train_csv":      data_dir / "train.csv",
        "val_csv":        data_dir / "val.csv",
        "test_csv":       data_dir / "test.csv",
        "summary_json":   data_dir / "summary.json",
        "run_dir":        run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "figures_dir":    run_dir / "figures",
        "results_csv_dir":run_dir / "results_csv",
        "generated_dir":  run_dir / "generated_audios",
    }

    if mkdir:
        for key in ("checkpoint_dir", "figures_dir", "results_csv_dir", "generated_dir"):
            paths[key].mkdir(parents=True, exist_ok=True)

    return paths
