from pathlib import Path

import torch

from .model import build_model


def load_detector(path, *, device='cpu'):
    """只加载可信队友交付的本项目 checkpoint；不下载外部权重。"""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = build_model(checkpoint['config'], checkpoint['metadata'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint
