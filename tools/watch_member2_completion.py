"""Keep progress current and finalize only after the existing queue exits cleanly."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_member2 import ROOT


def main():
    os.chdir(ROOT)
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUTF8='1')
    lock = Path('runs/member2/.runner.lock')
    last_count = -1
    while lock.exists():
        count = sum(len(path.read_text(encoding='utf-8').splitlines())
                    for path in Path('runs/member2').glob('*_seed*/metrics.jsonl'))
        if count != last_count:
            subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.member2_status'],
                           env=environment, stdout=subprocess.DEVNULL, check=True)
            print(f'Completed epochs: {count}/180', flush=True)
            last_count = count
        time.sleep(30)
    if Path('runs/member2/PAUSE_AFTER_CURRENT').exists():
        print('Queue paused; final delivery not started.', flush=True)
        return
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.finalize_member2'],
                   env=environment, check=True)


if __name__ == '__main__':
    main()
