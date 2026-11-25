"""Independent development-set evaluation; holdout is deliberately not opened here."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from cvdetect.engine import source_fingerprint
from cvdetect.inference import load_detector


CONDITIONS = ('clean', 'jpeg90', 'jpeg70', 'jpeg50', 'jpeg30', 'scale05')


def classification_metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('expected nonempty paired vectors')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('invalid labels/probabilities')
    predicted = (p >= .5).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y.astype(int), predicted), 1)
    per_class = []
    for c in (0, 1):
        per_class.append(dict(label=c, precision=float(cm[c, c]/cm[:, c].sum()) if cm[:, c].sum() else None,
                              recall=float(cm[c, c]/cm[c].sum()) if cm[c].sum() else None))
    # Average ranks handle tied scores exactly, without a third-party metric implementation.
    order = np.argsort(p, kind='stable')
    sorted_p = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ends = np.r_[starts[1:], len(p)]
    ranks = np.empty(len(p), dtype=float)
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + 1 + b) / 2
    positive = int(y.sum())
    negative = len(y) - positive
    auc = float((ranks[y == 1].sum()-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None
    return dict(count=len(y), accuracy=float((predicted == y).mean()), roc_auc=auc,
                confusion_matrix=cm.tolist(), per_class=per_class, threshold=.5)


def paired_group_bootstrap(left, right, *, iterations=10000, seed=17):
    """Accuracy(left)-Accuracy(right), sampling whole groups with replacement."""
    a = {r['image_id']: r for r in left}
    b = {r['image_id']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or not a or a.keys() != b.keys():
        raise ValueError('paired predictions must have unique identical image IDs')
    grouped = {}
    for identity, row in a.items():
        other = b[identity]
        if str(row['group_id']) != str(other['group_id']) or int(row['label']) != int(other['label']):
            raise ValueError('paired label/group mismatch')
        diff = int((float(row['prob_fake']) >= .5) == bool(int(row['label'])))
        diff -= int((float(other['prob_fake']) >= .5) == bool(int(other['label'])))
        grouped.setdefault(str(row['group_id']), []).append(diff)
    sums = np.array([sum(v) for _, v in sorted(grouped.items())])
    counts = np.array([len(v) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        idx = rng.integers(0, len(sums), len(sums))
        samples.append(sums[idx].sum()/counts[idx].sum())
    return dict(difference=float(sums.sum()/counts.sum()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                groups=len(sums), images=len(a), iterations=iterations, seed=seed)


def image_views(source, *, size, mode, patches, condition):
    image = ImageOps.exif_transpose(source).convert('RGB')
    if mode == 'global':
        if patches != 1:
            raise ValueError('global baseline only supports one view')
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    elif mode == 'local':
        rgb = np.asarray(image)
        dy, dx = max(0, size-rgb.shape[0]), max(0, size-rgb.shape[1])
        if dy or dx:
            image = Image.fromarray(np.pad(rgb, ((dy//2, dy-dy//2), (dx//2, dx-dx//2), (0, 0)), mode='reflect'))
    else:
        raise ValueError('unknown data mode')
    w, h = image.size
    positions = [((w-size)//2, (h-size)//2)]
    if patches == 5:
        positions += [(0, 0), (w-size, 0), (0, h-size), (w-size, h-size)]
    # JPEG is applied to the padded full image before fixed cropping, like team_data.
    if condition.startswith('jpeg'):
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=int(condition[4:]), subsampling=0)
        with Image.open(io.BytesIO(buffer.getvalue())) as encoded:
            image = encoded.convert('RGB')
    views = []
    for x, y in positions:
        patch = image.crop((x, y, x+size, y+size))
        if condition == 'scale05':
            patch = patch.resize((size//2, size//2), Image.Resampling.BILINEAR).resize((size, size), Image.Resampling.BILINEAR)
        views.append(torch.from_numpy(np.array(patch, copy=True)).permute(2, 0, 1).float()/255)
    return torch.stack(views)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def evaluate(checkpoint_path, output, *, device='cuda', patches=1, condition='clean', root='data/official'):
    output = Path(output)
    if output.exists():
        raise ValueError('choose a new output directory; evaluation evidence is not overwritten')
    model, ckpt = load_detector(checkpoint_path, device=device)
    config, metadata = ckpt['config'], ckpt['metadata']
    manifest = Path('data/splits.csv')
    if metadata['data_kind'] != 'official' or metadata['split_sha256'] != sha(manifest) or ckpt['source_sha256'] != source_fingerprint():
        raise ValueError('checkpoint/source/frozen split mismatch')
    with manifest.open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'dev']
    expected = metadata['splits']['dev']
    if [r['image_id'] for r in rows] != expected['image_ids'] or [r['group_id'] for r in rows] != expected['group_ids']:
        raise ValueError('dev inventory mismatch')
    predictions = []
    root = Path(root).resolve()
    batch_size = config['training']['batch_size']
    for start in range(0, len(rows), batch_size):
        members = rows[start:start+batch_size]
        images = []
        for row in members:
            path = (root/row['relative_path']).resolve()
            if not path.is_relative_to(root) or 'test' in Path(row['relative_path']).parts:
                raise ValueError('invalid dev path')
            with Image.open(path) as source:
                images.append(image_views(source, size=config['data']['image_size'], mode=config['data']['mode'],
                                          patches=patches, condition=condition))
        probabilities = model(torch.cat(images).to(device)).float().sigmoid().reshape(len(members), patches).mean(1).cpu().tolist()
        predictions.extend(dict(image_id=row['image_id'], group_id=row['group_id'], label=int(row['label']), prob_fake=p)
                           for row, p in zip(members, probabilities))
    metrics = classification_metrics([r['label'] for r in predictions], [r['prob_fake'] for r in predictions])
    report = dict(metrics=metrics, split='dev', condition=condition, patches=patches, aggregation='mean_probability',
                  checkpoint_sha256=sha(checkpoint_path), source_sha256=source_fingerprint(),
                  evaluator_sha256=sha(__file__), manifest_sha256=sha(manifest), config=config,
                  torch_version=str(torch.__version__), device=device, batch_size=batch_size,
                  holdout_evaluated=False, test_pixels_decoded=False)
    output.mkdir(parents=True)
    with (output/'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (output/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(metrics), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--root', default='data/official')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--patches', type=int, choices=[1, 5], default=1)
    parser.add_argument('--condition', choices=CONDITIONS, default='clean')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, root=args.root, device=args.device, patches=args.patches, condition=args.condition)


if __name__ == '__main__':
    main()
