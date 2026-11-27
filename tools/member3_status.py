"""Report E/F progress from saved epochs without loading model weights."""
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parents[1]
    runs, seconds = [], []
    for seed in (17, 29, 43):
        for experiment in ('E', 'F'):
            path = root/f'runs/member3/{experiment}_seed{seed}/metrics.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
            seconds.extend(row['seconds'] for row in rows)
            runs.append(dict(experiment=experiment, seed=seed, epochs=len(rows),
                             best_dev_accuracy=max((r['dev_accuracy'] for r in rows), default=None)))
    completed = sum(r['epochs'] for r in runs)
    report = dict(epochs_completed=completed, epochs_planned=120,
                  completed_runs=sum(r['epochs'] == 20 for r in runs), runs=runs,
                  median_epoch_seconds=statistics.median(seconds) if seconds else None,
                  approximate_remaining_hours=(120-completed)*statistics.median(seconds)/3600 if seconds else None,
                  timing_note='rough estimate from saved epoch wall time; excludes final replay and evaluation')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
