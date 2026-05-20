"""
Boucle d'entraînement DiffWave.
Extrait du notebook 50k steps, cellules cell-26 et cell-27.
Fix AMP : utilise torch.amp (API non-dépréciée) avec fallback CPU automatique.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .losses import stft_loss
from .plotting import save_loss_curves


class Trainer:
    def __init__(self, model, optimizer, schedule, cfg: dict, paths: dict,
                 device: str, summary: dict):
        self.model     = model
        self.optimizer = optimizer
        self.schedule  = schedule
        self.cfg       = cfg
        self.paths     = paths
        self.device    = device
        self.summary   = summary

        self.use_amp = cfg.get("use_amp", True) and device.startswith("cuda")
        self.scaler  = (
            torch.amp.GradScaler("cuda") if self.use_amp else None
        )

        self.global_step  = 0
        self.best_val_loss = float("inf")
        self.train_losses  = []
        self.val_losses    = []

    # ----------------------------------------------------------
    # Boucle principale
    # ----------------------------------------------------------
    def train(self, train_loader, val_loader) -> None:
        max_steps        = self.cfg["max_steps"]
        log_every        = self.cfg["log_every"]
        valid_every      = self.cfg["valid_every"]
        save_every       = self.cfg["save_every"]
        grad_clip        = self.cfg["grad_clip"]
        stft_weight      = self.cfg["stft_weight"]
        num_diff_steps   = self.schedule.num_steps

        self.model.train()

        pbar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
        )

        while self.global_step < max_steps:
            for batch in train_loader:
                audio = batch["audio"].to(self.device)
                mel   = batch["mel"].to(self.device)

                diffusion_steps = torch.randint(
                    low=0, high=num_diff_steps,
                    size=(audio.shape[0],), device=self.device,
                )
                noisy_audio, noise = self.schedule.add_noise(audio, diffusion_steps)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        predicted_noise = self.model(noisy_audio, diffusion_steps, mel)
                        if predicted_noise.dim() == 3:
                            predicted_noise = predicted_noise.squeeze(1)
                        mse  = F.mse_loss(predicted_noise, noise)
                        stft = stft_loss(predicted_noise.float(), noise.float())
                        loss = mse + stft_weight * stft

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    predicted_noise = self.model(noisy_audio, diffusion_steps, mel)
                    if predicted_noise.dim() == 3:
                        predicted_noise = predicted_noise.squeeze(1)
                    mse  = F.mse_loss(predicted_noise, noise)
                    stft = stft_loss(predicted_noise, noise)
                    loss = mse + stft_weight * stft

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                self.global_step += 1
                self.train_losses.append(loss.item())

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mse":  f"{mse.item():.4f}",
                    "stft": f"{stft.item():.4f}",
                })

                if self.global_step % valid_every == 0:
                    pbar.clear()
                    val_loss = self._validate(val_loader)
                    self.val_losses.append({"step": self.global_step, "val_loss": val_loss})

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best", self.global_step)
                        tqdm.write(f"[Step {self.global_step}] Val loss: {val_loss:.6f} ✓ best checkpoint saved")
                    else:
                        tqdm.write(f"[Step {self.global_step}] Val loss: {val_loss:.6f}")

                if self.global_step % save_every == 0:
                    self._save_checkpoint("last", self.global_step)

                if self.global_step >= max_steps:
                    break

        pbar.close()

        # Checkpoint final + figures
        self._save_checkpoint("final", self.global_step)
        save_loss_curves(
            self.train_losses, self.val_losses,
            self.paths["figures_dir"],
            self.cfg["experiment_name"],
        )
        print("Entraînement terminé.")

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------
    def _validate(self, val_loader) -> float:
        max_batches = self.cfg.get("valid_max_batches", 20)
        stft_weight = self.cfg["stft_weight"]
        num_diff_steps = self.schedule.num_steps

        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(self.device)
                mel   = batch["mel"].to(self.device)

                diffusion_steps = torch.randint(
                    low=0, high=num_diff_steps,
                    size=(audio.shape[0],), device=self.device,
                )
                noisy_audio, noise = self.schedule.add_noise(audio, diffusion_steps)

                predicted_noise = self.model(noisy_audio, diffusion_steps, mel)
                if predicted_noise.dim() == 3:
                    predicted_noise = predicted_noise.squeeze(1)

                mse  = F.mse_loss(predicted_noise, noise)
                stft = stft_loss(predicted_noise, noise)
                loss = mse + stft_weight * stft

                total_loss += loss.item()
                n_batches  += 1

                if n_batches >= max_batches:
                    break

        self.model.train()
        return total_loss / max(1, n_batches)

    # ----------------------------------------------------------
    # Sauvegarde / chargement de checkpoint
    # ----------------------------------------------------------
    def _save_checkpoint(self, tag: str, step: int) -> None:
        """tag : 'best' | 'last' | 'final'"""
        checkpoint_dir = self.paths["checkpoint_dir"]
        path = checkpoint_dir / f"{tag}.pt"
        torch.save({
            "step":                step,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "train_losses":        self.train_losses,
            "val_losses":          self.val_losses,
            "cfg":                 self.cfg,
            "summary":             self.summary,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, model, optimizer, device: str) -> dict:
        """
        Charge un checkpoint et met à jour model + optimizer.
        Retourne le dict complet du checkpoint.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt
