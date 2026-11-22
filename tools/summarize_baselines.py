"""成员1 G/A开发集结果审计与汇总；不评价holdout/test。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
import torch

from cvdetect.engine import evaluate_dev, seed_everything, source_fingerprint
from cvdetect.inference import load_detector
from team_data import build_loaders


def inspect_run(directory, *, expected_epochs=20):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    history = [json.loads(line) for line in (directory/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    if [r['epoch'] for r in history] != list(range(1, expected_epochs+1)) or config['training']['epochs'] != expected_epochs:
        raise ValueError(f'incomplete epoch sequence: {directory}')
    if config['status'] != 'frozen_budget' or metadata['data_kind'] != 'official':
        raise ValueError(f'not a frozen official run: {directory}')
    for row in history:
        if (row['experiment'], row['seed'], row['data_kind']) != (config['experiment'], config['seed'], 'official'):
            raise ValueError('log identity disagrees with config')
        if any(not math.isfinite(row[key]) for key in ('train_loss', 'train_accuracy', 'dev_loss', 'dev_accuracy', 'seconds')):
            raise ValueError('nonfinite training metrics')
    chosen = max(history, key=lambda r: (r['dev_accuracy'], -r['dev_loss']))
    fingerprint = source_fingerprint()
    for name, epoch in (('best.pt', chosen['epoch']), ('last.pt', expected_epochs)):
        checkpoint = torch.load(directory/name, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['metadata'] != metadata or checkpoint['source_sha256'] != fingerprint:
            raise ValueError(f'checkpoint provenance mismatch: {name}')
        if checkpoint['epoch'] != epoch or checkpoint['metrics'] != history[epoch-1]:
            raise ValueError(f'checkpoint epoch/selection mismatch: {name}')
    with (directory/'dev_predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    ids = [r['image_id'] for r in predictions]
    members = metadata['splits']['dev']
    membership = dict(zip(members['image_ids'], members['group_ids']))
    if len(ids) != len(set(ids)) or set(ids) != set(membership) or any(membership[r['image_id']] != r['group_id'] for r in predictions):
        raise ValueError('dev prediction inventory/group mismatch')
    probabilities = [float(r['prob_fake']) for r in predictions]
    labels = [int(r['label']) for r in predictions]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or any(y not in (0, 1) for y in labels):
        raise ValueError('invalid dev probabilities/labels')
    accuracy = sum((p >= 0.5) == bool(y) for p, y in zip(probabilities, labels))/len(labels)
    if accuracy != chosen['dev_accuracy']:
        raise ValueError('prediction accuracy disagrees with selected checkpoint')
    hashes = {}
    for name in ('config.json', 'metadata.json', 'environment.json', 'metrics.jsonl', 'best.pt', 'last.pt', 'dev_predictions.csv'):
        with (directory/name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {'experiment': config['experiment'], 'seed': config['seed'], 'data_kind': 'official',
              'best_epoch': chosen['epoch'], 'dev_accuracy': accuracy, 'dev_bce': chosen['dev_loss'],
              'last_dev_accuracy': history[-1]['dev_accuracy'], 'dev_count': len(predictions),
              'parameters': chosen['parameters'], 'recorded_epoch_seconds_total': sum(r['seconds'] for r in history),
              'recorded_epoch_seconds_median': statistics.median(r['seconds'] for r in history),
              'recorded_epoch_seconds_min': min(r['seconds'] for r in history),
              'recorded_epoch_seconds_max': max(r['seconds'] for r in history),
              'peak_vram_bytes': max(r['peak_vram_bytes'] for r in history), 'source_sha256': fingerprint,
              'split_sha256': metadata['split_sha256'], 'data_revision': metadata.get('data_revision'),
              'run_directory': directory.as_posix(), 'artifact_sha256': hashes}
    return result, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs/baselines')
    parser.add_argument('--output', default='results/baselines')
    parser.add_argument('--verify-predictions', action='store_true', help='reload each best.pt and rerun clean dev only')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--experiments', nargs='+', default=['G', 'A'],
                        help='要汇总的实验组；组号需与 configs/<组>.json 对应')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    authorization = json.loads(Path('validation/baseline_execution_authorization.json').read_text(encoding='utf-8'))
    manifest = list(csv.DictReader(Path('data/splits.csv').read_text(encoding='utf-8').splitlines()))
    dev_labels = {r['image_id']: int(r['label']) for r in manifest if r['split'] == 'dev'}
    records, histories = [], {}
    for seed in args.seeds:
        for experiment in args.experiments:
            directory = Path(args.runs)/f'{experiment}_seed{seed}'
            record, history = inspect_run(directory)
            config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
            expected = json.loads(Path(f'configs/{experiment}.json').read_text(encoding='utf-8'))
            expected.update(seed=seed, adapter='team_data:build_loaders', adapter_source_sha256=authorization['data_adapter_sha256'])
            if config != expected or record['split_sha256'] != authorization['split_sha256'] or record['source_sha256'] != authorization['source_sha256']:
                raise ValueError('run differs from authorized frozen experiment')
            with (directory/'dev_predictions.csv').open(encoding='utf-8', newline='') as stream:
                saved = {r['image_id']: r for r in csv.DictReader(stream)}
            if set(saved) != set(dev_labels) or any(int(saved[i]['label']) != y for i, y in dev_labels.items()):
                raise ValueError('dev labels differ from frozen manifest')
            if args.verify_predictions:
                seed_everything(seed)
                bundle = build_loaders(config, seed=seed)
                model, checkpoint = load_detector(directory/'best.pt', device=args.device)
                if checkpoint['metadata'] != bundle.metadata:
                    raise ValueError('current data pipeline differs from checkpoint metadata')
                metrics, replay = evaluate_dev(model, bundle.dev_loader, torch.device(args.device))
                errors = [abs(float(saved[r['image_id']]['prob_fake'])-r['prob_fake']) for r in replay]
                if metrics['accuracy'] != record['dev_accuracy'] or not math.isclose(metrics['loss'], record['dev_bce'], rel_tol=1e-6, abs_tol=1e-6) or max(errors) > 1e-6:
                    raise ValueError(f'checkpoint dev replay mismatch: {directory}')
                record['checkpoint_replay'] = {'passed': True, 'images': len(replay), 'max_probability_error': max(errors), 'bce': metrics['loss'], 'device': args.device}
                del model, bundle
            records.append(record)
            histories[experiment, seed] = history
            print(f'verified {experiment} seed={seed}: best epoch {record["best_epoch"]}, dev {record["dev_accuracy"]:.6f}', flush=True)
    summaries = []
    for experiment in args.experiments:
        subset = [r for r in records if r['experiment'] == experiment]
        summaries.append({'experiment': experiment, 'seeds': '/'.join(str(s) for s in args.seeds),
                          'runs': len(args.seeds),
                          'dev_accuracy_mean': statistics.mean(r['dev_accuracy'] for r in subset),
                          'dev_accuracy_sample_std': statistics.stdev(r['dev_accuracy'] for r in subset),
                          'dev_bce_mean': statistics.mean(r['dev_bce'] for r in subset),
                          'dev_bce_sample_std': statistics.stdev(r['dev_bce'] for r in subset),
                          'recorded_epoch_seconds_total': sum(r['recorded_epoch_seconds_total'] for r in subset)})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('per_run.csv', [{k: v for k, v in r.items() if k not in ('artifact_sha256', 'checkpoint_replay')} for r in records]), ('summary.csv', summaries)):
        with (output/name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {'passed': True, 'scope': 'clean single-patch dev, selected on this same dev; no independent generalization claim',
              'completed_runs': len(records),
              'epochs_per_run': min(len(h) for h in histories.values()), 'threshold': 0.5, 'checkpoint_selection': 'dev accuracy, lower BCE, earlier epoch',
              'checkpoint_predictions_replayed': args.verify_predictions, 'holdout_evaluated': False, 'test_pixels_decoded': False,
              'recorded_seconds_exclude_checkpoint_io_and_process_startup': True, 'runs': records, 'summary': summaries,
              'timing_caveat': 'perf_counter epoch wall time may include machine suspension; retain raw values and inspect median/min/max; not GPU compute time',
              }
    by_experiment = {item['experiment']: item for item in summaries}
    if 'A' in by_experiment and 'G' in by_experiment:
        report['A_minus_G_mean_percentage_points'] = 100*(by_experiment['A']['dev_accuracy_mean']-by_experiment['G']['dev_accuracy_mean'])
    (output/'baseline_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']
    for index, experiment in enumerate(args.experiments):
        color = palette[index % len(palette)]
        for split, style in (('train', '--'), ('dev', '-')):
            for axis, metric in zip(axes, ('loss', 'accuracy')):
                values = np.array([[r[f'{split}_{metric}'] for r in histories[experiment, seed]] for seed in args.seeds])
                mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
                epochs = np.arange(1, len(mean) + 1)
                axis.plot(epochs, mean, style, color=color, label=f'{experiment} {split}')
                if split == 'dev':
                    axis.fill_between(epochs, mean-std, mean+std, color=color, alpha=0.12)
    for axis, title in zip(axes, ('Loss', 'Accuracy')):
        axis.set(xlabel='Epoch', ylabel=title, xlim=(1, max(len(h) for h in histories.values())),
                 title=title + f' (mean of {len(args.seeds)} seeds)')
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output/'training_curves.png', dpi=240)
    figure.savefig(output/'training_curves.pdf')
    plt.close(figure)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
