from __future__ import annotations

from pathlib import Path

import torch


def unwrap_state_dict(payload):
    """兼容 raw state_dict 和 {'model': state_dict} 两种 checkpoint。"""

    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def load_weights_flexible(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, int]:
    """尽量加载能匹配的权重，便于复用旧 N2N checkpoint。"""

    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = unwrap_state_dict(payload)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model_state = model.state_dict()

    loaded = 0
    skipped = 0
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    return {"loaded": loaded, "skipped": skipped}


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    epoch: int,
    args,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "epoch": int(epoch),
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)
