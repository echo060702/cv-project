"""Audit one completed B/C/D run without claiming that all nine are finished."""
import argparse
import json
import math
import os
from pathlib import Path

import torch

from cvdetect.engine import evaluate_dev, seed_everything
from cvdetect.inference import load_detector
from team_data import build_loaders
from tools.run_member2 import ROOT, freeze_execution
from tools.summarize_baselines import inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    freeze = freeze_execution()
    directory = Path(args.run)
    record, history = inspect_run(directory)
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    experiment, seed = config['experiment'], config['seed']
    if experiment not in freeze['experiments'] or seed not in freeze['seeds']:
        raise ValueError('not a member2 frozen experiment/seed')
    expected = json.loads(Path(f'configs/{experiment}.json').read_text(encoding='utf-8'))
    expected.update(seed=seed, adapter='team_data:build_loaders', adapter_source_sha256=freeze['data_adapter_sha256'])
    if config != expected:
        raise ValueError('run config differs from execution freeze')
    seed_everything(seed)
    bundle = build_loaders(config, seed=seed)
    model, checkpoint = load_detector(directory/'best.pt', device='cuda')
    reference = json.loads(Path(f'runs/baselines/A_seed{seed}/metadata.json').read_text(encoding='utf-8'))
    if checkpoint['metadata'] != bundle.metadata or bundle.metadata != reference:
        raise ValueError('data metadata differs from delivered A')
    metrics, predictions = evaluate_dev(model, bundle.dev_loader, torch.device('cuda'))
    import csv
    with (directory/'dev_predictions.csv').open(encoding='utf-8', newline='') as stream:
        saved = {r['image_id']: r for r in csv.DictReader(stream)}
    if any(int(saved[r['image_id']]['label']) != r['label'] for r in predictions):
        raise ValueError('saved dev labels differ from current frozen loader')
    error = max(abs(float(saved[r['image_id']]['prob_fake'])-r['prob_fake']) for r in predictions)
    if metrics['accuracy'] != record['dev_accuracy'] or not math.isclose(metrics['loss'], record['dev_bce'], abs_tol=1e-6, rel_tol=1e-6) or error > 1e-6:
        raise ValueError('checkpoint dev replay mismatch')
    record.update(passed=True, checkpoint_replay=dict(passed=True, images=len(predictions),
                  max_probability_error=error, dev_bce=metrics['loss']),
                  scope='one completed run; development checkpoint-selection set, not independent test',
                  holdout_evaluated=False, test_pixels_decoded=False)
    output = Path('results/member2')
    output.mkdir(parents=True, exist_ok=True)
    (output/f'{experiment}_seed{seed}_audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for axis, metric in zip(axes, ('loss', 'accuracy')):
        for split in ('train', 'dev'):
            axis.plot(range(1, 21), [r[f'{split}_{metric}'] for r in history], label=split)
        axis.set(xlabel='Epoch', ylabel=metric, title=f'{experiment} seed {seed}', xlim=(1, 20))
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output/f'{experiment}_seed{seed}_curves.png', dpi=180)
    plt.close(fig)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
