"""
Multi-resolution STFT loss (extrait du notebook 50k steps, cellule cell-23).
"""
import torch
import torch.nn.functional as F


def stft_loss(x: torch.Tensor, y: torch.Tensor,
              fft_sizes=(512, 1024, 2048),
              hop_sizes=(128, 256, 512)) -> torch.Tensor:
    """
    Calcule la perte STFT multi-résolution (L1 sur log-magnitude).

    x, y : Tensor [B, T]
    """
    loss = 0.0
    for fft_size, hop_size in zip(fft_sizes, hop_sizes):
        window = torch.hann_window(fft_size, device=x.device)
        X = torch.stft(x, fft_size, hop_size, return_complex=True, window=window)
        Y = torch.stft(y, fft_size, hop_size, return_complex=True, window=window)
        loss += F.l1_loss(
            X.abs().clamp(min=1e-8).log(),
            Y.abs().clamp(min=1e-8).log()
        )
    return loss / len(fft_sizes)
