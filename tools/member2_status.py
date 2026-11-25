"""Show real B/C/D progress using completed epoch logs only."""
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parents[1]
    rows = []
    for seed in (17, 29, 43):
        for experiment in ('B', 'C', 'D'):
            directory = root/f'runs/member2/{experiment}_seed{seed}'
            path = directory/'metrics.jsonl'
            history = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
            best = max(history, key=lambda r: (r['dev_accuracy'], -r['dev_loss'])) if history else None
            row = dict(experiment=experiment, seed=seed, epochs_completed=len(history), epochs_planned=20,
                       status='complete' if len(history) == 20 else 'partial' if history else
                       'initialized_no_completed_epoch' if (directory/'config.json').exists() else 'not_started',
                       best_dev_accuracy=best['dev_accuracy'] if best else None,
                       best_epoch=best['epoch'] if best else None,
                       last_epoch_seconds=history[-1]['seconds'] if history else None)
            rows.append(row)
    completed = sum(r['epochs_completed'] for r in rows)
    seconds = [r['last_epoch_seconds'] for r in rows if r['last_epoch_seconds'] is not None]
    report = dict(completed_runs=sum(r['status'] == 'complete' for r in rows), planned_runs=9,
                  completed_epochs=completed, planned_epochs=180, runs=rows,
                  estimated_remaining_hours=(180-completed)*statistics.median(seconds)/3600 if seconds else None,
                  estimate_caveat='rough estimate from finished epochs; excludes download, verification and checkpoint I/O',
                  final_refit_status='waiting for member3 selection summary and evaluation freeze')
    target = root/'results/member2/progress.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
