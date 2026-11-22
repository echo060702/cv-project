"""成员1公共训练器：仅 train/dev，按 epoch 恢复，最终确认评估由成员3负责。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import validate_bundle, validate_config
from .losses import paired_loss
from .model import build_model


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def train_epoch(model, loader, optimizer, scaler, device, amp, epoch, training):
    model.train()
    totals = {'loss': 0., 'supervised': 0., 'js': 0., 'correct': 0, 'count': 0}
    ramp_steps = max(1, math.ceil(training['epochs'] * len(loader) * training['warmup_fraction']))
    for step, batch in enumerate(loader):
        normal = batch['image'].to(device)
        jpeg = batch['image_jpeg'].to(device)
        labels = batch['label'].to(device).float()
        coefficient = training['consistency'] * min(1., (epoch*len(loader)+step+1)/ramp_steps)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits_a, logits_b = model(torch.cat((normal, jpeg))).chunk(2)
            loss, parts = paired_loss(logits_a, logits_b, labels, coefficient)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = len(labels)
        totals['count'] += n
        totals['correct'] += ((logits_a.detach() >= 0) == labels.bool()).sum().item()
        for key, value in (('loss', loss.detach()), ('supervised', parts['supervised']), ('js', parts['js'])):
            totals[key] += value.item()*n
    return {**{key: totals[key]/totals['count'] for key in ('loss', 'supervised', 'js')},
            'accuracy': totals['correct']/totals['count'], 'count': totals['count'], 'coefficient': coefficient}


@torch.inference_mode()
def evaluate_dev(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    rows = []
    for batch in loader:
        image, labels = batch['image'].to(device), batch['label'].to(device).float()
        logits = model(image).float()
        probs = logits.sigmoid()
        count += len(labels)
        loss_sum += F.binary_cross_entropy_with_logits(logits, labels, reduction='sum').item()
        correct += ((probs >= 0.5) == labels.bool()).sum().item()
        rows.extend({'image_id': i, 'group_id': g, 'label': int(y), 'prob_fake': float(p)}
                    for i, g, y, p in zip(batch['image_id'], batch['group_id'], labels.cpu(), probs.cpu()))
    return {'loss': loss_sum/count, 'accuracy': correct/count, 'count': count}, rows


def run_training(config, bundle, output, *, device='cpu', resume=None, run_epochs=None):
    validate_config(config)
    validate_bundle(bundle, config)
    if bundle.metadata['data_kind'] == 'official' and config['status'] != 'frozen_budget':
        raise ValueError('official runs require status=frozen_budget after timing and shared budget freeze')
    output, device = Path(output), torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; select --device cpu explicitly')
    training = config['training']
    if run_epochs is not None and run_epochs < 1:
        raise ValueError('run_epochs must be positive')
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError('output is not empty; choose a new directory or --resume last.pt')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config['seed'])
    model = build_model(config, bundle.metadata).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training['epochs'])
    amp = bool(training['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    fingerprint = source_fingerprint()
    best_key, start = (-1., -float('inf')), 0
    if resume is not None:
        resume = Path(resume)
        if resume.resolve() != (output/'last.pt').resolve():
            raise ValueError('resume must use the same output directory last.pt')
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['metadata'] != bundle.metadata:
            raise ValueError('resume config/metadata differs; do not mix experiments')
        if checkpoint['source_sha256'] != fingerprint:
            raise ValueError('resume source changed; create a new run')
        history = [json.loads(line) for line in (output/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        if [r['epoch'] for r in history] != list(range(1, checkpoint['epoch']+1)):
            raise ValueError('log/checkpoint epochs disagree; retain artifacts and reconcile before resume')
        if not (output/'best.pt').is_file():
            raise ValueError('best.pt missing from resumed run')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best_key = checkpoint['epoch'], tuple(checkpoint['best_key'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        random.setstate(checkpoint['rng']['python'])
        state = checkpoint['rng']['numpy']
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
                   'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'amp': amp, 'deterministic_algorithms': True}
    for filename, payload in (('config.json', config), ('metadata.json', bundle.metadata), ('environment.json', environment)):
        if resume is None:
            (output/filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    stop = min(training['epochs'], start+run_epochs) if run_epochs else training['epochs']
    for epoch in range(start, stop):
        bundle.set_epoch(epoch)
        model.train()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        lr_used = optimizer.param_groups[0]['lr']
        train = train_epoch(model, bundle.train_loader, optimizer, scaler, device, amp, epoch, training)
        dev, predictions = evaluate_dev(model, bundle.dev_loader, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        key = (dev['accuracy'], -dev['loss'])
        improved = key > best_key
        if improved:
            best_key = key
        scheduler.step()
        row = {'epoch': epoch+1, 'experiment': config['experiment'], 'seed': config['seed'],
               'data_kind': bundle.metadata['data_kind'], 'train_loss': train['loss'],
               'train_supervised': train['supervised'], 'train_js': train['js'],
               'train_accuracy': train['accuracy'], 'dev_loss': dev['loss'], 'dev_accuracy': dev['accuracy'],
               'learning_rate': lr_used, 'consistency_weight': train['coefficient'], 'seconds': time.perf_counter()-began,
               'peak_vram_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
               'parameters': sum(p.numel() for p in model.parameters()), 'best': improved}
        numpy_state = np.random.get_state()
        checkpoint = {'format_version': 1, 'epoch': epoch+1, 'config': config, 'metadata': bundle.metadata,
                      'source_sha256': fingerprint, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'best_key': best_key,
                      'metrics': row, 'environment': environment,
                      'rng': {'torch': torch.get_rng_state(), 'python': random.getstate(),
                              'numpy': (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
                              'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
        if improved:
            atomic_save(checkpoint, output/'best.pt')
            with (output/'dev_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['image_id', 'group_id', 'label', 'prob_fake'])
                writer.writeheader()
                writer.writerows(predictions)
        atomic_save(checkpoint, output/'last.pt')
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return output/'best.pt'
