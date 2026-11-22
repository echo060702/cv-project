"""冻结后的最终有标签数据重训，不使用开发/保留集选 checkpoint。"""
import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time

import torch

from .contracts import validate_config, validate_metadata
from .engine import atomic_save, seed_everything, source_fingerprint, train_epoch
from .model import build_model


def run_refit(config, loader, metadata, set_epoch, recipe, output, *, device='cuda'):
    config = copy.deepcopy(config)
    config['training']['epochs'] = recipe['epochs']
    validate_config(config)
    validate_metadata(metadata, refit=True)
    if recipe['selected_experiment'] != config['experiment']:
        raise ValueError('recipe selected_experiment differs from config')
    for key in ('selection_summary_sha256', 'evaluation_freeze_sha256'):
        if len(recipe[key]) != 64 or any(c not in '0123456789abcdef' for c in recipe[key]):
            raise ValueError('recipe requires selection/evaluation freeze SHA256')
    if metadata['data_kind'] == 'official' and config['status'] != 'frozen_budget':
        raise ValueError('official refit requires frozen_budget')
    output, device = Path(output), torch.device(device)
    if len(loader) == 0 or (output.exists() and any(output.iterdir())):
        raise ValueError('refit needs nonempty loader and empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config['seed'])
    training = config['training']
    model = build_model(config, metadata).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training['epochs'])
    amp = bool(training['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for name, payload in (('config.json', config), ('metadata.json', metadata), ('recipe.json', recipe)):
        (output/name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    for epoch in range(training['epochs']):
        set_epoch(epoch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        train = train_epoch(model, loader, optimizer, scaler, device, amp, epoch, training)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        scheduler.step()
        row = {'epoch': epoch+1, 'experiment': config['experiment'], 'seed': config['seed'],
               'data_kind': metadata['data_kind'], 'purpose': 'final_refit_no_validation',
               'train_loss': train['loss'], 'train_accuracy': train['accuracy'], 'learning_rate': lr,
               'train_supervised': train['supervised'], 'train_js': train['js'],
               'seconds': time.perf_counter()-began, 'parameters': sum(p.numel() for p in model.parameters()),
               'peak_vram_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0}
        checkpoint = {'format_version': 1, 'purpose': 'final_refit_no_validation', 'epoch': epoch+1,
                      'config': config, 'metadata': metadata, 'recipe': recipe, 'metrics': row,
                      'source_sha256': source_fingerprint(), 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict()}
        atomic_save(checkpoint, output/'final.pt')
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return output/'final.pt'


def main():
    parser = argparse.ArgumentParser(description='仅在正式评估冻结之后进行全量有标签重训')
    for name in ('config', 'adapter', 'recipe', 'selection-summary', 'evaluation-freeze', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    recipe = json.loads(Path(args.recipe).read_text(encoding='utf-8'))
    for key, path in (('selection_summary_sha256', args.selection_summary), ('evaluation_freeze_sha256', args.evaluation_freeze)):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != recipe[key]:
            raise ValueError(f'{key} does not match delivered file')
    module, function = args.adapter.split(':', 1)
    factory = getattr(importlib.import_module(module), function)
    config['adapter'] = args.adapter
    config['adapter_source_sha256'] = hashlib.sha256(Path(inspect.getfile(factory)).read_bytes()).hexdigest()
    bundle = factory(config, seed=config['seed'])
    run_refit(config, bundle.train_loader, bundle.metadata, bundle.set_epoch, recipe, args.output, device=args.device)


if __name__ == '__main__':
    main()
