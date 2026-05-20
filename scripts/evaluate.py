"""
Comparaison de plusieurs runs DiffWave (courbes de loss, métriques).
Correspond au notebook 04_evaluation_resultas.ipynb.

Usage :
    python scripts/evaluate.py \
        --runs runs/diffwave_doda_scratch runs/diffwave_doda_finetune \
        --labels "From scratch 50k" "Fine-tune 10k" \
        --output_dir runs/evaluation
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation et comparaison des runs DiffWave")
    parser.add_argument("--runs",       nargs="+", required=True,
                        help="Répertoires des runs à comparer")
    parser.add_argument("--labels",     nargs="+", default=None,
                        help="Étiquettes pour chaque run (défaut : nom du répertoire)")
    parser.add_argument("--output_dir", default="runs/evaluation",
                        help="Répertoire de sortie pour les figures")
    return parser.parse_args()


# ============================================================
# Chargement des métriques d'un run
# ============================================================
def load_run_metrics(run_dir: Path) -> dict:
    """Charge training_losses.csv et validation_losses.csv depuis results_csv/."""
    csv_dir = run_dir / "results_csv"

    train_path = csv_dir / "training_losses.csv"
    val_path   = csv_dir / "validation_losses.csv"

    train_df = pd.read_csv(train_path) if train_path.exists() else None
    val_df   = pd.read_csv(val_path)   if val_path.exists()   else None

    return {"train": train_df, "val": val_df}


# ============================================================
# Figures comparatives
# ============================================================
def plot_comparison(runs_data: list, labels: list, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Courbes d'entraînement ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for metrics, label in zip(runs_data, labels):
        if metrics["train"] is not None:
            df = metrics["train"]
            ax.plot(df["step"], df["train_loss"], label=label, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train MSE Loss")
    ax.set_title("Comparaison des courbes d'entraînement")
    ax.legend()
    ax.grid(True)
    fig.savefig(output_dir / "comparison_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_dir / 'comparison_training_loss.png'}")

    # --- Courbes de validation ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for metrics, label in zip(runs_data, labels):
        if metrics["val"] is not None:
            df = metrics["val"]
            ax.plot(df["step"], df["val_loss"], marker="o", label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation MSE Loss")
    ax.set_title("Comparaison des courbes de validation")
    ax.legend()
    ax.grid(True)
    fig.savefig(output_dir / "comparison_validation_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_dir / 'comparison_validation_loss.png'}")


def print_summary_table(runs_data: list, labels: list) -> None:
    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 60)
    rows = []
    for metrics, label in zip(runs_data, labels):
        row = {"Run": label}

        if metrics["train"] is not None:
            df = metrics["train"]
            row["Steps total"]     = int(df["step"].max())
            row["Train loss final"] = round(float(df["train_loss"].iloc[-1]), 6)
            row["Train loss min"]   = round(float(df["train_loss"].min()), 6)

        if metrics["val"] is not None:
            df = metrics["val"]
            row["Val loss min"] = round(float(df["val_loss"].min()), 6)
            best_step = int(df.loc[df["val_loss"].idxmin(), "step"])
            row["Meilleur step val"] = best_step

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    print()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    run_dirs = [Path(r) for r in args.runs]
    labels   = args.labels if args.labels else [r.name for r in run_dirs]
    output_dir = Path(args.output_dir)

    if len(labels) != len(run_dirs):
        print("Erreur : --labels doit avoir le même nombre d'éléments que --runs")
        sys.exit(1)

    # Chargement
    runs_data = []
    for run_dir, label in zip(run_dirs, labels):
        print(f"Chargement de : {run_dir} ({label})")
        metrics = load_run_metrics(run_dir)
        runs_data.append(metrics)

    # Affichage résumé
    print_summary_table(runs_data, labels)

    # Figures
    plot_comparison(runs_data, labels, output_dir)

    # Export CSV de comparaison
    rows = []
    for metrics, label in zip(runs_data, labels):
        if metrics["val"] is not None:
            df = metrics["val"].copy()
            df["run"] = label
            rows.append(df)
    if rows:
        comparison_df = pd.concat(rows, ignore_index=True)
        comparison_df.to_csv(output_dir / "comparison_val_losses.csv", index=False)
        print(f"CSV comparaison : {output_dir / 'comparison_val_losses.csv'}")

    print(f"\nÉvaluation terminée. Résultats dans : {output_dir}")


if __name__ == "__main__":
    main()
