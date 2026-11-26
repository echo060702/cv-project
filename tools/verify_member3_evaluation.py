"""Verify published dev predictions and the frozen choice without GPU inference."""
import argparse
import json
import os
from pathlib import Path
import statistics

from cvdetect.engine import source_fingerprint
from team_eval import CONDITIONS, sha
from tools.run_member3_evaluation import OUTPUT, ROOT, cached


def verify(check_checkpoints=False):
    protocol = json.loads((OUTPUT/'protocol.json').read_bytes())
    freeze = json.loads(Path('results/evaluation_freeze.json').read_bytes())
    selection = json.loads(Path('results/selection_summary.json').read_bytes())
    for key, path in [('protocol_sha256', OUTPUT/'protocol.json'),
                      ('selection_summary_sha256', 'results/selection_summary.json'),
                      ('evaluator_sha256', 'team_eval.py'),
                      ('runner_sha256', 'tools/run_member3_evaluation.py'),
                      ('split_sha256', 'data/splits.csv'),
                      ('statistics_sha256', 'data/statistics.json')]:
        if freeze[key] != sha(path):
            raise ValueError(f'Frozen {key} mismatch')
    if freeze['source_sha256'] != source_fingerprint():
        raise ValueError('Frozen shared source mismatch')
    for key, freeze_key in [('manifest_sha256', 'split_sha256'),
                            ('statistics_sha256', 'statistics_sha256'),
                            ('evaluator_sha256', 'evaluator_sha256'),
                            ('source_sha256', 'source_sha256')]:
        if protocol[key] != freeze[freeze_key]:
            raise ValueError(f'Protocol {key} mismatch')
    expected = {f'{exp}_seed{seed}_p{patches}_{condition}'
                for exp in 'GABCDEF' for seed in (17, 29, 43)
                for patches, condition in [(1, c) for c in CONDITIONS]
                + ([(5, 'clean')] if exp != 'G' else [])}
    if len(protocol['jobs']) != 144 or {j['key'] for j in protocol['jobs']} != expected:
        raise ValueError('Incomplete evaluation matrix')
    reports = {j['key']: cached(j, protocol)[0] for j in protocol['jobs']}
    candidates = []
    for exp in 'GABCDEF':
        for patches in ([1] if exp == 'G' else [1, 5]):
            values = [reports[f'{exp}_seed{s}_p{patches}_clean'] for s in (17, 29, 43)]
            candidates.append(dict(experiment=exp, patches=patches,
                parameters=2795297 if values[0]['config']['model']['branch'] == 'none' else 2888897,
                mean=statistics.mean(r['metrics']['accuracy'] for r in values),
                sample_std=statistics.stdev(r['metrics']['accuracy'] for r in values),
                mean_wall_seconds=statistics.mean(r['wall_seconds'] for r in values)))
    selected = sorted(candidates, key=lambda c: (-c['mean'], c['patches'], c['parameters'], c['experiment']))[0]
    if candidates != selection['candidates'] or selected != selection['selected']:
        raise ValueError('Saved candidate ranking mismatch')
    if (selected['experiment'], selected['patches'], .5) != (freeze['selected_experiment'], freeze['patches'], freeze['threshold']):
        raise ValueError('Frozen choice mismatch')
    if check_checkpoints:
        for job in freeze['checkpoints']:
            if sha(job['checkpoint']) != job['checkpoint_sha256']:
                raise ValueError(f"Checkpoint mismatch: {job['checkpoint']}")
    return dict(dev_jobs_verified=len(reports), predictions_recomputed=144*1200,
                selected=selected, checkpoint_hashes_verified=21 if check_checkpoints else 0,
                gpu_inference_rerun=False, holdout_evaluated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-checkpoints', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    print(json.dumps(verify(args.check_checkpoints), indent=2))


if __name__ == '__main__':
    main()
