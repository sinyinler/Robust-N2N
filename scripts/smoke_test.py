from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses.ntn_losses import ExplicitNoiseTranslationLoss
from models.denoiser import Denoiser
from models.ntn import NoiseTranslator


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)
    translator = NoiseTranslator(input_channels=1, width=8, middle_blocks=1, inject_sigma=0.5)
    gaussian_expert = Denoiser(input_channels=1)
    criterion = ExplicitNoiseTranslationLoss(beta=2e-3)

    translated = translator(x)
    denoised = gaussian_expert(translated)
    loss, spatial, freq = criterion(translated - x)

    assert translated.shape == x.shape, translated.shape
    assert denoised.shape == x.shape, denoised.shape
    assert torch.isfinite(loss), loss
    print(
        "smoke ok | "
        f"translated={tuple(translated.shape)} denoised={tuple(denoised.shape)} "
        f"explicit={float(loss.detach()):.6f} "
        f"spatial={float(spatial.detach()):.6f} freq={float(freq.detach()):.6f}"
    )


if __name__ == "__main__":
    main()
