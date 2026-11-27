"""One clean holdout confirmation of the already frozen method; no selection."""
import argparse
import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

from cvdetect.inference import load_detector
from team_eval import classification_metrics, image_views, sha
from tools.verify_member3_evaluation import verify

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('results/holdout_confirmation')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def validate_inventory(rows, metadata):
    expected = metadata['splits']['holdout']
    if len(rows) != 1200 or len({r['image_id'] for r in rows}) != 1200:
        raise ValueError('Holdout inventory must contain 1200 unique images')
    if [r['image_id'] for r in rows] != expected['image_ids'] or [r['group_id'] for r in rows] != expected['group_ids']:
        raise ValueError('Holdout checkpoint inventory mismatch')
    ids = {r['image_id'] for r in rows}
    groups = {r['group_id'] for r in rows}
    for split in ('train', 'dev'):
        if ids.intersection(metadata['splits'][split]['image_ids']) or groups.intersection(metadata['splits'][split]['group_ids']):
            raise ValueError('Holdout overlaps train/dev')


def read_predictions(folder, rows, protocol_hash):
    report = json.loads((folder/'metrics.json').read_bytes())
    if report['protocol_sha256'] != protocol_hash or report['predictions_sha256'] != sha(folder/'predictions.csv'):
        raise ValueError('Confirmation evidence checksum mismatch')
    with (folder/'predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    inventory = [(r['image_id'], r['group_id'], int(r['label'])) for r in rows]
    if [(r['image_id'], r['group_id'], int(r['label'])) for r in predictions] != inventory:
        raise ValueError('Saved confirmation inventory mismatch')
    metrics = classification_metrics([int(r['label']) for r in predictions], [float(r['prob_fake']) for r in predictions])
    if metrics != report['metrics']:
        raise ValueError('Saved confirmation metrics mismatch')
    return report


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    verify(check_checkpoints=True)
    freeze_path = Path('results/evaluation_freeze.json')
    freeze = json.loads(freeze_path.read_bytes())
    if freeze['threshold'] != .5 or freeze['aggregation'] != 'mean_probability':
        raise ValueError('Unsupported frozen rule')
    jobs = [j for j in freeze['checkpoints'] if j['experiment'] == freeze['selected_experiment']]
    if [j['seed'] for j in jobs] != freeze['seeds']:
        raise ValueError('Frozen seed inventory mismatch')
    with Path('data/splits.csv').open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['split'] == 'holdout']
    with Path('data/audit/images.csv').open(encoding='utf-8', newline='') as stream:
        audit = {r['image_id']: r for r in csv.DictReader(stream)}
    image_root = Path('data/official').resolve()
    for row in rows:
        path = (image_root/row['relative_path']).resolve()
        if not path.is_relative_to(image_root) or 'test' in path.parts or sha(path) != audit[row['image_id']]['file_sha256']:
            raise ValueError('Holdout file provenance mismatch')
    protocol = dict(version=1, scope='selected method only; clean holdout; all frozen seeds; no tuning',
        evaluation_freeze_sha256=sha(freeze_path), selection_summary_sha256=sha('results/selection_summary.json'),
        evaluator_sha256=sha('team_eval.py'), runner_sha256=sha(__file__),
        audit_sha256=sha('data/audit/images.csv'), split_sha256=sha('data/splits.csv'),
        jobs=jobs, seeds=freeze['seeds'], condition='clean', patches=freeze['patches'], threshold=.5,
        aggregation=freeze['aggregation'], count=len(rows), device='cpu', batch_size=16,
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        numpy_version=np.__version__, pillow_version=pillow_version,
        rule='Report every frozen seed, mean and sample std; no seed/epoch/threshold/model reselection',
        timing='protocol saved before holdout model inference', test_pixels_decoded=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_bytes()) != protocol:
            raise ValueError('Confirmation protocol changed; preserve the existing evidence')
    else:
        if args.verify_only:
            raise ValueError('No existing confirmation')
        write_json(protocol_path, protocol)
    if args.prepare_only:
        print('Prepared 3 frozen-seed clean holdout jobs; no model inference.', flush=True)
        return
    protocol_hash = sha(protocol_path)
    complete = OUTPUT/'summary.json'
    reports = []
    for job in jobs:
        folder = OUTPUT/f"{job['experiment']}_seed{job['seed']}"
        if folder.exists():
            report = read_predictions(folder, rows, protocol_hash)
        else:
            if args.verify_only or complete.exists():
                raise ValueError('Confirmation incomplete; will not infer during verification')
            model, checkpoint = load_detector(job['checkpoint'], device='cpu')
            config, metadata = checkpoint['config'], checkpoint['metadata']
            validate_inventory(rows, metadata)
            if config['experiment'] != job['experiment'] or config['seed'] != job['seed'] or checkpoint['source_sha256'] != freeze['source_sha256'] or metadata['split_sha256'] != freeze['split_sha256'] or metadata['statistics_sha256'] != freeze['statistics_sha256']:
                raise ValueError('Checkpoint provenance mismatch')
            # Exclusive creation prevents accidental duplicate confirmation jobs.
            folder.mkdir()
            write_json(folder/'started.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=protocol_hash))
            predictions = []
            began = time.perf_counter()
            for start in range(0, len(rows), 16):
                batch = rows[start:start+16]
                images = []
                for row in batch:
                    with Image.open(image_root/row['relative_path']) as source:
                        images.append(image_views(source, size=config['data']['image_size'], mode=config['data']['mode'], patches=freeze['patches'], condition='clean'))
                probabilities = model(torch.cat(images)).float().sigmoid().reshape(len(batch), freeze['patches']).mean(1).tolist()
                predictions.extend(dict(image_id=r['image_id'], group_id=r['group_id'], label=int(r['label']), prob_fake=p) for r, p in zip(batch, probabilities))
                if (start+len(batch)) % 400 == 0:
                    print(f"Seed {job['seed']}: {start+len(batch)}/{len(rows)}", flush=True)
            with (folder/'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
                writer.writeheader(); writer.writerows(predictions)
            report = dict(seed=job['seed'], experiment=job['experiment'], split='holdout', condition='clean',
                checkpoint_sha256=job['checkpoint_sha256'], protocol_sha256=protocol_hash,
                predictions_sha256=sha(folder/'predictions.csv'), wall_seconds=time.perf_counter()-began,
                metrics=classification_metrics([r['label'] for r in predictions], [r['prob_fake'] for r in predictions]),
                holdout_evaluated=True, test_pixels_decoded=False)
            write_json(folder/'metrics.json', report)
            report = read_predictions(folder, rows, protocol_hash)
        reports.append(report)
        print(json.dumps(dict(seed=job['seed'], metrics=report['metrics'])), flush=True)
    summary = dict(status='complete', experiment=freeze['selected_experiment'], patches=freeze['patches'], threshold=.5,
        protocol_sha256=protocol_hash, evaluation_freeze_sha256=sha(freeze_path), reports=reports,
        accuracy_mean=statistics.mean(r['metrics']['accuracy'] for r in reports),
        accuracy_sample_std=statistics.stdev(r['metrics']['accuracy'] for r in reports),
        roc_auc_mean=statistics.mean(r['metrics']['roc_auc'] for r in reports),
        unique_images=len(rows), prediction_rows=len(rows)*len(reports),
        independent_observation_note='Same 1200 images across seeds; not 3600 independent samples',
        method_changed_after_confirmation=False, holdout_evaluated=True, test_pixels_decoded=False,
        environment_note='CPU PyTorch 2.5.1; prior dev evaluation used CUDA PyTorch 2.7.1; no retuning')
    if complete.exists():
        if json.loads(complete.read_bytes()) != summary:
            raise ValueError('Saved summary mismatch')
    else:
        if args.verify_only:
            raise ValueError('Summary missing')
        write_json(complete, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
