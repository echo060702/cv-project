"""Run the nine frozen B/C/D experiments serially, without rerunning G/A."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from cvdetect.contracts import validate_config
from cvdetect.engine import source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
EXPERIMENTS = ('B', 'C', 'D')


def freeze_execution(root=ROOT):
    """Compare against delivered recipes before recording execution provenance."""
    root = Path(root)
    baseline = json.loads((root/'runs/baselines/A_seed17/config.json').read_text(encoding='utf-8'))
    configs = {}
    for experiment in EXPERIMENTS:
        path = root/f'configs/{experiment}.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        validate_config(config)
        expected = {k: v for k, v in baseline.items() if k not in ('adapter', 'adapter_source_sha256')}
        expected = json.loads(json.dumps(expected))
        expected['experiment'] = experiment
        expected['model']['branch'] = 'none' if experiment == 'C' else 'residual'
        expected['training']['consistency'] = 0. if experiment == 'B' else 0.1
        if config != expected or config['status'] != 'frozen_budget':
            raise ValueError(f'{experiment}: recipe differs from the delivered common budget')
        configs[experiment] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = json.loads((root/'validation/member2_data_pipeline.json').read_text(encoding='utf-8'))
    if not report['passed'] or not report['statistics_recomputed_train_only']:
        raise ValueError('restore and verify official data including train-only statistics first')
    metadata = json.loads((root/'runs/baselines/A_seed17/metadata.json').read_text(encoding='utf-8'))
    state = dict(source_sha256=source_fingerprint(),
                 data_adapter_sha256=hashlib.sha256((root/'team_data.py').read_bytes()).hexdigest(),
                 split_sha256=hashlib.sha256((root/'data/splits.csv').read_bytes()).hexdigest(),
                 statistics_sha256=hashlib.sha256((root/'data/statistics.json').read_bytes()).hexdigest(),
                 config_sha256=configs, experiments=list(EXPERIMENTS), seeds=list(SEEDS),
                 epochs=20, initialization='random', selection='dev accuracy, lower BCE, earlier epoch',
                 threshold=0.5, holdout_evaluated=False, test_pixels_decoded=False)
    if state['source_sha256'] != json.loads((root/'results/baselines/baseline_summary.json').read_text(encoding='utf-8'))['runs'][0]['source_sha256']:
        raise ValueError('shared model/training source differs from delivered G/A')
    revision = 'paired-rgb-v1-' + hashlib.sha256((root/'team_data.py').read_bytes() +
                                                 (root/'tools/prepare_data.py').read_bytes() +
                                                 (root/'data/statistics.json').read_bytes()).hexdigest()
    if state['data_adapter_sha256'] != baseline['adapter_source_sha256'] or revision != metadata['data_revision']:
        raise ValueError('data adapter/revision differs from delivered G/A')
    if state['split_sha256'] != metadata['split_sha256'] or state['statistics_sha256'] != metadata['statistics_sha256']:
        raise ValueError('data split/statistics differ from delivered G/A')
    target = root/'validation/member2_execution_freeze.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != state:
            raise ValueError('execution inputs changed since freeze; preserve runs and reconcile')
    else:
        target.write_text(json.dumps(state, indent=2), encoding='utf-8')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume-queue', action='store_true', help='explicitly clear a requested queue pause')
    args = parser.parse_args()
    freeze_execution()
    if args.preflight_only:
        print('Frozen budget/source/data verified; nine runs planned.', flush=True)
        return
    output = ROOT/'runs/member2'
    output.mkdir(parents=True, exist_ok=True)
    pause = output/'PAUSE_AFTER_CURRENT'
    if pause.exists():
        if args.resume_queue:
            pause.unlink()
        else:
            print('Queue is paused; use --resume-queue only when ready to continue.', flush=True)
            return
    # Exclusive creation prevents two orchestrators from sharing a run directory.
    lock = output/'.runner.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), started=datetime.now().isoformat()), stream)
    environment = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        for seed in SEEDS:
            for experiment in EXPERIMENTS:
                run = output/f'{experiment}_seed{seed}'
                arguments = [sys.executable, '-X', 'utf8', '-m', 'cvdetect.train',
                             '--config', f'configs/{experiment}.json', '--adapter', 'team_data:build_loaders',
                             '--seed', str(seed), '--output', str(run), '--device', 'cuda']
                metrics = run/'metrics.jsonl'
                if metrics.exists():
                    history = [json.loads(line) for line in metrics.read_text(encoding='utf-8').splitlines()]
                    if [row['epoch'] for row in history] != list(range(1, len(history)+1)) or len(history) > 20:
                        raise ValueError(f'{run}: invalid epoch sequence')
                    if len(history) == 20:
                        from tools.summarize_baselines import inspect_run
                        inspect_run(run)
                        print(f'verified completed {experiment} seed={seed}; skipping', flush=True)
                        continue
                if (run/'last.pt').exists():
                    arguments += ['--resume', str(run/'last.pt')]
                print(f'starting {experiment} seed={seed}', flush=True)
                # Inherit stdout so the caller sees each finished epoch and any error.
                subprocess.run(arguments, cwd=ROOT, env=environment, check=True)
                print(f'finished {experiment} seed={seed}', flush=True)
                if pause.exists():
                    print('Pause requested; current run completed, no next run started.', flush=True)
                    return
        subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.summarize_member2', '--verify-predictions'],
                       cwd=ROOT, env=environment, check=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
