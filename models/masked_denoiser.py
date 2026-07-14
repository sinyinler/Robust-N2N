# -*- coding: utf-8 -*-
"""Mask-aware wrapper around the existing lightweight U-Net.

The denoiser receives two channels during training:
  1) the (possibly masked) log-domain BFI image;
  2) a binary visibility mask (1=visible, 0=hidden).

At inference ``visible_mask`` is optional.  Omitting it supplies an all-one
mask, so callers can continue to invoke ``model(image)``.
"""
from __future__ import annotations

import torch

from models.denoiser_feats import DenoiserWithFeats, FEAT_CHANNELS


class MaskedDenoiserWithFeats(DenoiserWithFeats):
    """Two-channel denoiser with an inference-compatible one-image API."""

    def __init__(self, image_channels: int = 1):
        image_channels = int(image_channels)
        super().__init__(input_channels=image_channels + 1)
        self.image_channels = image_channels

    def _prepare_input(
        self,
        image: torch.Tensor,
        visible_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != self.image_channels:
            raise ValueError(
                f"image must have shape (N,{self.image_channels},H,W), got {tuple(image.shape)}"
            )
        if visible_mask is None:
            visible_mask = torch.ones(
                (image.shape[0], 1, image.shape[2], image.shape[3]),
                device=image.device,
                dtype=image.dtype,
            )
        if visible_mask.ndim != 4 or visible_mask.shape[1] != 1:
            raise ValueError(f"visible_mask must have shape (N,1,H,W), got {tuple(visible_mask.shape)}")
        if visible_mask.shape[0] != image.shape[0] or visible_mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                f"visible_mask shape {tuple(visible_mask.shape)} is incompatible with image {tuple(image.shape)}"
            )
        visible_mask = visible_mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
        return torch.cat((image, visible_mask), dim=1)

    def forward(
        self,
        image: torch.Tensor,
        visible_mask: torch.Tensor | None = None,
        return_feats: bool = False,
    ):
        x = self._prepare_input(image, visible_mask)
        return super().forward(x, return_feats=return_feats)


__all__ = ["MaskedDenoiserWithFeats", "FEAT_CHANNELS"]
