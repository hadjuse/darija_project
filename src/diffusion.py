"""
Schedule de diffusion DiffWave.
Encapsule les variables globales du notebook (noise_schedule, alphas, alpha_cumprod)
dans une classe réutilisable — extrait des cellules cell-22 et cell-37.
"""
import torch
from tqdm.auto import tqdm


class DiffusionSchedule:
    """
    Pré-calcule et stocke les tenseurs du schedule de diffusion.
    Fournit add_noise, p_sample et generate.
    """

    def __init__(self, noise_schedule_list: list, device: str):
        self.device = device
        self.noise_schedule = torch.tensor(
            noise_schedule_list, dtype=torch.float32, device=device
        )
        self.alphas = 1.0 - self.noise_schedule
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_steps = len(noise_schedule_list)

    # ----------------------------------------------------------
    # Diffusion avant : ajoute du bruit à l'audio propre
    # ----------------------------------------------------------
    def add_noise(self, clean_audio: torch.Tensor, diffusion_steps: torch.Tensor):
        """
        clean_audio : [B, T]
        diffusion_steps : [B]  (entiers longs)
        Retourne (noisy_audio [B,T], noise [B,T])
        """
        noise = torch.randn_like(clean_audio)

        sqrt_alpha_cumprod = torch.sqrt(
            self.alpha_cumprod[diffusion_steps]
        ).unsqueeze(1)
        sqrt_one_minus = torch.sqrt(
            1.0 - self.alpha_cumprod[diffusion_steps]
        ).unsqueeze(1)

        noisy_audio = sqrt_alpha_cumprod * clean_audio + sqrt_one_minus * noise
        return noisy_audio, noise

    # ----------------------------------------------------------
    # Diffusion inverse : une étape de débruitage
    # ----------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, step: int, mel: torch.Tensor):
        """
        x   : [B, T]
        mel : [B, 80, F]
        step : int (index du pas, de NUM_STEPS-1 à 0)
        Retourne x_prev [B, T]
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        batch_size = x.shape[0]
        t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)

        predicted_noise = model(x, t, mel)

        if predicted_noise.dim() == 3:
            predicted_noise = predicted_noise.squeeze(1)

        beta_t          = self.noise_schedule[step]
        alpha_t         = self.alphas[step]
        alpha_cumprod_t = self.alpha_cumprod[step]

        coef = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * predicted_noise)

        if step > 0:
            sigma  = torch.sqrt(beta_t)
            x_prev = mean + sigma * torch.randn_like(x)
        else:
            x_prev = mean

        return x_prev

    # ----------------------------------------------------------
    # Génération complète depuis un Mel spectrogram
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, model, mel: torch.Tensor, audio_length: int) -> torch.Tensor:
        """
        Génère un audio par diffusion inverse.

        mel : [B, 80, F] ou [80, F] (sera unsqueezed si nécessaire)
        audio_length : nombre de samples à générer
        Retourne audio [B, T] normalisé en [-1, 1]
        """
        model.eval()

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)

        x = torch.randn(mel.shape[0], audio_length, device=self.device)

        for step in tqdm(
            reversed(range(self.num_steps)),
            total=self.num_steps,
            desc="Génération",
            leave=False,
        ):
            x = self.p_sample(model, x, step, mel)

        # Normalisation finale
        x = x / (x.abs().max(dim=1, keepdim=True)[0] + 1e-8)
        return x
