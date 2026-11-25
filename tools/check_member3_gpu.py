"""工程验收，明确不产生 Kaggle 数据成绩。"""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import PIL
import torch

from cvdetect.engine import run_training, seed_everything, source_fingerprint
from cvdetect.inference import load_detector
from cvdetect.losses import paired_loss
from cvdetect.model import Detector
from tests.fixtures.synthetic_adapter import build_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='validation/member3_gpu_smoke')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
                   'pillow': PIL.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
                   'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                   'source_sha256': source_fingerprint(), 'scope': 'synthetic engineering only'}
    report = {'environment': environment, 'full_size_gradient': [], 'synthetic_fits': []}
    for experiment in ('E', 'F'):
        config = json.loads((root/'configs'/f'{experiment}.json').read_text(encoding='utf-8'))
        seed_everything(17)
        model = Detector(width=32, branch=config['model']['branch']).to(device)
        paired_batch_size = 16 if device == 'cuda' else 2
        x = torch.rand(paired_batch_size, 3, 224, 224, device=device)
        labels = (torch.arange(paired_batch_size, device=device) % 2).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
            a, b = model(torch.cat((x, x))).chunk(2)
            loss, _ = paired_loss(a, b, labels, config['training']['consistency'])
        scaler.scale(loss).backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        valid = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        scaler.step(optimizer)
        scaler.update()
        if device == 'cuda':
            torch.cuda.synchronize()
        result = {'experiment': experiment, 'image_size': 224, 'width': 32, 'paired_batch_size': paired_batch_size,
                  'parameters': sum(p.numel() for p in model.parameters()), 'finite_gradients': valid,
                  'optimizer_step_ok': True,
                  'seconds': time.perf_counter()-began,
                  'peak_vram_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0}
        report['full_size_gradient'].append(result)
        if not valid:
            raise AssertionError(f'{experiment}: invalid gradients')
        del model, x, a, b, loss, optimizer, scaler
        if device == 'cuda':
            torch.cuda.empty_cache()
        small = copy.deepcopy(config)
        small['status'] = 'synthetic_engineering_only'
        small['model']['width'] = 4
        small['data']['image_size'] = 32
        small['training'].update(epochs=6, batch_size=8, learning_rate=0.003)
        folder = output/f'synthetic_{experiment}'
        checkpoint = run_training(small, build_loaders(small, seed=17), folder, device=device)
        loaded, state = load_detector(checkpoint, device=device)
        sample = next(iter(build_loaders(small, seed=17).dev_loader))
        with torch.inference_mode():
            p = loaded(sample['image'].to(device)).sigmoid()
        if not torch.isfinite(p).all() or state['metrics']['train_accuracy'] < 0.95:
            raise AssertionError(f'{experiment}: tiny synthetic fitting check failed')
        report['synthetic_fits'].append({'experiment': experiment, 'best_epoch': state['epoch'],
                                         'train_accuracy': state['metrics']['train_accuracy'],
                                         'dev_accuracy': state['metrics']['dev_accuracy'],
                                         'data_kind': state['metadata']['data_kind'],
                                         'checkpoint_load_ok': True})
    report['passed'] = True
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
