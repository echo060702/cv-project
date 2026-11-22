"""第二轮：从实际ZIP独立解压、校验、测试和CLI训练/恢复。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from .package_team import PACKAGE_ROOT, verify_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--with-baselines', action='store_true', help='audit and replay all six delivered official G/A checkpoints')
    parser.add_argument('--with-progress', action='store_true', help='recompute the bundled handoff progress snapshot and compare')
    args = parser.parse_args()
    archive, workdir, report_path = Path(args.archive).resolve(), Path(args.workdir).resolve(), Path(args.report).resolve()
    if workdir.exists() and any(workdir.iterdir()):
        raise ValueError('use an empty check workdir')
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise AssertionError('ZIP CRC check failed')
        for name in package.namelist():
            if not (workdir/name).resolve().is_relative_to(workdir):
                raise ValueError('ZIP path escapes target')
        package.extractall(workdir)
    root = workdir/PACKAGE_ROOT
    file_count = verify_manifest(root)
    environment = dict(os.environ, PYTHONUTF8='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    environment.pop('PYTHONPATH', None)
    calls = [
        ('import_location', ['-c', 'import cvdetect,team_data,json; print(json.dumps([cvdetect.__file__,team_data.__file__]))']),
        ('real_data_handoff', ['-m', 'tools.verify_data_handoff', '--output', str(workdir/'received_data_check.json')]),
        ('unit_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v']),
        ('cli_training', ['-m', 'cvdetect.train', '--config', 'configs/B.json', '--adapter', 'tests.fixtures.synthetic_adapter:build_loaders',
                          '--output', str(workdir/'cli_run'), '--epochs', '4', '--run-epochs', '2', '--batch-size', '8', '--width', '4', '--image-size', '32', '--device', 'cpu']),
        ('cli_resume', ['-m', 'cvdetect.train', '--config', 'configs/B.json', '--adapter', 'tests.fixtures.synthetic_adapter:build_loaders',
                        '--output', str(workdir/'cli_run'), '--epochs', '4', '--batch-size', '8', '--width', '4', '--image-size', '32', '--device', 'cpu',
                        '--resume', str(workdir/'cli_run'/'last.pt')]),
    ]
    if args.with_progress:
        calls.insert(2, ('progress_recheck', ['-m', 'tools.progress_report', '--output', str(workdir/'progress_recheck')]))
    if args.with_baselines:
        calls.insert(2, ('formal_baselines', ['-m', 'tools.summarize_baselines', '--runs', 'runs/baselines',
                      '--output', str(workdir/'baseline_recheck'), '--verify-predictions', '--device', 'cuda']))
    result = {'round': 2, 'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
              'verified_payload_files': file_count, 'zip_crc_valid': True, 'python': sys.version,
              'uses_existing_environment': True, 'independent_extracted_workdir': str(root), 'checks': []}
    for name, arguments in calls:
        completed = subprocess.run([sys.executable, '-X', 'utf8', *arguments], cwd=root, env=environment,
                                   capture_output=True, text=True, encoding='utf-8')
        log_path = report_path.parent/f'round2_{name}.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout+completed.stderr, encoding='utf-8')
        if completed.returncode != 0:
            raise AssertionError(f'{name} failed: {log_path}')
        if name == 'import_location' and not all(Path(p).is_relative_to(root) for p in json.loads(completed.stdout)):
            raise AssertionError('import used original workspace instead of extracted package')
        result['checks'].append({'name': name, 'passed': True, 'log': log_path.name})
    if args.with_progress:
        bundled = json.loads((root/'results/handoff/progress.json').read_text(encoding='utf-8'))
        recomputed = json.loads((workdir/'progress_recheck/progress.json').read_text(encoding='utf-8'))
        assert bundled['completed_runs'] == recomputed['completed_runs']
        assert [(r['run'], r['epochs_done'], r['best_epoch'], round(r['best_dev_accuracy'], 9)) for r in bundled['runs']] == \
               [(r['run'], r['epochs_done'], r['best_epoch'], round(r['best_dev_accuracy'], 9)) for r in recomputed['runs']]
        assert bundled['pending'] == recomputed['pending']
    history = [json.loads(line) for line in (workdir/'cli_run'/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['epoch'] for row in history] == [1, 2, 3, 4]
    assert all(row['data_kind'] == 'synthetic_engineering' for row in history)
    assert verify_manifest(root) == file_count
    result.update(passed=True, cli_epochs_after_resume=[1, 2, 3, 4], original_workspace_import_used=False,
                  fresh_dependency_install_tested=False, official_training_tested=False,
                  official_baseline_artifacts_and_checkpoint_replay_verified=args.with_baselines,
                  handoff_progress_snapshot_reproduced=args.with_progress,
                  official_training_reexecuted_in_round2=False)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
