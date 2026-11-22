"""中途交接用：按当前 runs/baselines 实际状态生成逐种子进度与当前最优准确率。

与 tools/summarize_baselines.py 的区别：本工具允许未跑满 20 epoch 的 run 存在，
只如实报告已完成轮数、当前最优 checkpoint 与是否可从 last.pt 继续，不产出正式结论。
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

EXPERIMENTS = ('G', 'A')
SEEDS = (17, 29, 43)
EXPECTED_EPOCHS = 20


def read_progress(directory):
    metrics = directory/'metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'empty metrics: {metrics}')
    epochs = [row['epoch'] for row in rows]
    if epochs != list(range(1, len(epochs)+1)):
        raise ValueError(f'metrics epochs are not contiguous 1..N: {directory}')
    best = max(rows, key=lambda row: (row['dev_accuracy'], -row['dev_loss'], -row['epoch']))
    checkpoint = torch.load(directory/'last.pt', map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != epochs[-1]:
        raise ValueError(f'last.pt epoch {checkpoint["epoch"]} != logged {epochs[-1]}: {directory}')
    return {
        'run': directory.name,
        'experiment': rows[0]['experiment'],
        'seed': rows[0]['seed'],
        'data_kind': rows[0]['data_kind'],
        'epochs_done': len(rows),
        'epochs_expected': EXPECTED_EPOCHS,
        'complete': len(rows) == EXPECTED_EPOCHS,
        'last_checkpoint_epoch': checkpoint['epoch'],
        'resumable': True,
        'best_epoch': best['epoch'],
        'best_dev_accuracy': best['dev_accuracy'],
        'best_dev_loss': best['dev_loss'],
        'seconds_total': sum(row['seconds'] for row in rows),
        'peak_vram_bytes': max(row['peak_vram_bytes'] for row in rows),
        'parameters': rows[0]['parameters'],
        'artifacts': sorted(path.name for path in directory.iterdir() if path.is_file()),
    }


def summarize(values):
    count = len(values)
    mean = sum(values)/count
    variance = sum((value-mean)**2 for value in values)/(count-1) if count > 1 else float('nan')
    return {'count': count, 'mean': mean, 'std': math.sqrt(variance) if count > 1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='runs/baselines')
    parser.add_argument('--output', default='results/handoff')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs_root = root/args.runs
    output = root/args.output
    output.mkdir(parents=True, exist_ok=True)

    records, pending = [], []
    for seed in SEEDS:
        for experiment in EXPERIMENTS:
            directory = runs_root/f'{experiment}_seed{seed}'
            if not (directory/'metrics.jsonl').is_file() or not (directory/'last.pt').is_file():
                pending.append({'experiment': experiment, 'seed': seed, 'epochs_done': 0, 'status': 'not_started'})
                continue
            record = read_progress(directory)
            records.append(record)
            if not record['complete']:
                pending.append({'experiment': experiment, 'seed': seed, 'epochs_done': record['epochs_done'],
                                'status': 'partial_resumable'})

    complete = [record for record in records if record['complete']]
    aggregates = {}
    for experiment in EXPERIMENTS:
        values = [record['best_dev_accuracy'] for record in complete if record['experiment'] == experiment]
        aggregates[experiment] = summarize(values) if values else None

    report = {
        'generated_at': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        'source': 'tools/progress_report.py',
        'runs_root': args.runs,
        'scope_note': '当前进度快照：仅统计实际存在的 run；未跑满 20 epoch 的 run 不计入均值。不评价 holdout/test。',
        'selection_rule': 'dev 单 patch Accuracy 最高；并列取较低 dev BCE；再并列取较早 epoch；阈值 0.5',
        'completed_runs': len(complete),
        'total_planned_runs': len(EXPERIMENTS)*len(SEEDS),
        'runs': records,
        'pending': pending,
        'per_experiment_completed': aggregates,
        'all_runs_complete': len(complete) == len(EXPERIMENTS)*len(SEEDS),
    }
    (output/'progress.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    fields = ['run', 'experiment', 'seed', 'epochs_done', 'epochs_expected', 'complete',
              'best_epoch', 'best_dev_accuracy', 'best_dev_loss', 'seconds_total', 'peak_vram_bytes', 'resumable']
    with (output/'progress.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})
        for item in pending:
            if item['status'] == 'not_started':
                writer.writerow({'run': f"{item['experiment']}_seed{item['seed']}", 'experiment': item['experiment'],
                                 'seed': item['seed'], 'epochs_done': 0, 'epochs_expected': EXPECTED_EPOCHS,
                                 'complete': False, 'best_epoch': '', 'best_dev_accuracy': '', 'best_dev_loss': '',
                                 'seconds_total': '', 'peak_vram_bytes': '', 'resumable': False})
    print(json.dumps({'completed_runs': report['completed_runs'], 'all_runs_complete': report['all_runs_complete'],
                      'pending': pending,
                      'summary': [{'run': record['run'], 'epochs': record['epochs_done'],
                                   'best_epoch': record['best_epoch'],
                                   'best_dev_accuracy': round(record['best_dev_accuracy'], 6)} for record in records]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
