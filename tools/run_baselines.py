"""成员1入口：按固定种子顺序串行运行G/A。"""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    output = root/'runs/baselines'
    output.mkdir(parents=True, exist_ok=True)
    for seed in (17, 29, 43):
        for experiment in ('G', 'A'):
            run = output/f'{experiment}_seed{seed}'
            config = json.loads((root/f'configs/{experiment}.json').read_text(encoding='utf-8'))
            if config['status'] != 'frozen_budget':
                raise ValueError('freeze the shared budget before official baselines')
            arguments = [sys.executable, '-X', 'utf8', '-m', 'cvdetect.train', '--config', f'configs/{experiment}.json',
                         '--adapter', 'team_data:build_loaders', '--seed', str(seed), '--output', str(run), '--device', 'cuda']
            metrics = run/'metrics.jsonl'
            if metrics.is_file():
                done = len(metrics.read_text(encoding='utf-8').splitlines())
                if done >= config['training']['epochs']:
                    print(f'skipping {experiment} seed={seed}: already {done} epochs', flush=True)
                    continue
            if (run/'last.pt').is_file():
                arguments += ['--resume', str(run/'last.pt')]
            log = (output/f'{experiment}_seed{seed}_console.txt').open('a', encoding='utf-8')
            print(f'starting {experiment} seed={seed}', flush=True)
            code = subprocess.run(arguments, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
            log.close()
            print(f'finished {experiment} seed={seed} exit={code}', flush=True)
            if code != 0:
                raise RuntimeError(f'{experiment} seed={seed} failed; inspect console log before continuing')


if __name__ == '__main__':
    main()
