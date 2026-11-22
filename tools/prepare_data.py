"""成员1：训练图审计、重复分组、固定划分与仅train统计。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_images(root, labels, output):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(labels).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'image_id', 'label', 'image_path'}.issubset(rows[0]):
        raise ValueError('labels must contain image_id,label,image_path')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate image IDs in source labels')
    # DCT perceptual hash is only a candidate screen, never a learned feature.
    coordinate = np.arange(32)
    dct = np.cos(np.pi*(2*coordinate[None, :]+1)*coordinate[:, None]/64)
    dct[0] *= 1/np.sqrt(2)
    records = []
    for index, row in enumerate(rows):
        path = (root/row['image_path']).resolve()
        if not path.is_relative_to(root) or 'test' in Path(row['image_path']).parts or row['label'] not in ('0', '1'):
            raise ValueError(f'invalid labeled training path/label: {row}')
        record = {'image_id': row['image_id'], 'relative_path': Path(row['image_path']).as_posix(), 'label': int(row['label'])}
        try:
            content = path.read_bytes()
            with Image.open(path) as source:
                record.update(width=source.width, height=source.height, mode=source.mode, format=source.format,
                              exif_orientation=source.getexif().get(274, 1))
                image = ImageOps.exif_transpose(source).convert('RGB')
                pixels = np.asarray(image)
                record['file_sha256'] = hashlib.sha256(content).hexdigest()
                record['pixel_sha256'] = hashlib.sha256(str(image.size).encode()+pixels.tobytes()).hexdigest()
                gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
                low = (dct@gray@dct.T)[:8, :8].reshape(-1)
                bits = low > np.median(low[1:])
                record['phash'] = np.packbits(bits).tobytes().hex()
                record['status'] = 'ok'
                record['error'] = ''
        except (OSError, ValueError, UnidentifiedImageError) as error:
            record.update(status='error', error=str(error))
        records.append(record)
        if (index+1) % 500 == 0:
            print(f'audited {index+1}/{len(rows)} training images', flush=True)
    fields = ['image_id', 'relative_path', 'label', 'width', 'height', 'mode', 'format', 'exif_orientation',
              'file_sha256', 'pixel_sha256', 'phash', 'status', 'error']
    write_csv(output/'images.csv', records, fields)
    good = [r for r in records if r['status'] == 'ok']
    hashes = np.array([int(r['phash'], 16) for r in good], dtype=np.uint64)
    candidates = []
    for index in range(len(good)):
        distances = np.bitwise_count(hashes[:index] ^ hashes[index])
        for previous in np.flatnonzero(distances <= 4):
            a, b = good[int(previous)], good[index]
            if a['pixel_sha256'] != b['pixel_sha256']:
                candidates.append({'image_id_a': a['image_id'], 'image_id_b': b['image_id'],
                                   'hamming': int(distances[previous]), 'label_a': a['label'], 'label_b': b['label']})
    write_csv(output/'near_duplicate_candidates.csv', candidates, ['image_id_a', 'image_id_b', 'hamming', 'label_a', 'label_b'])
    print(json.dumps({'audited': len(records), 'errors': len(records)-len(good), 'near_candidates': len(candidates)}), flush=True)
    return records


def freeze_split(audit, output, *, seed=17, confirmed_pairs=None):
    if isinstance(audit, (str, Path)):
        with Path(audit).open(encoding='utf-8', newline='') as stream:
            audit = list(csv.DictReader(stream))
    if any(row['status'] != 'ok' for row in audit):
        raise ValueError('audit has unreadable files; resolve and document before splitting')
    audit = sorted(audit, key=lambda r: r['image_id'])
    parents = {r['image_id']: r['image_id'] for r in audit}

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    pixels = {}
    for row in audit:
        digest = row['pixel_sha256']
        if digest in pixels:
            union(row['image_id'], pixels[digest])
        else:
            pixels[digest] = row['image_id']
    if confirmed_pairs:
        with Path(confirmed_pairs).open(encoding='utf-8', newline='') as stream:
            for pair in csv.DictReader(stream):
                if pair['decision'] == 'same_family':
                    union(pair['image_id_a'], pair['image_id_b'])
    groups = {}
    for row in audit:
        groups.setdefault(find(row['image_id']), []).append(row)
    rng = np.random.default_rng(seed)
    assignment = {}
    counts = {split: [0, 0] for split in ('train', 'dev', 'holdout')}
    for label in (0, 1):
        families = [family for family in groups.values() if int(family[0]['label']) == label]
        if any(len({int(r['label']) for r in family}) != 1 for family in families):
            raise ValueError('duplicate family contains conflicting labels; resolve before splitting')
        # Shuffle then process largest groups first; choose largest remaining quota.
        rng.shuffle(families)
        families.sort(key=len, reverse=True)
        total = sum(map(len, families))
        target = np.array([0.70, 0.15, 0.15])*total
        current = np.zeros(3)
        for family in families:
            selected = int(np.argmax(target-current))
            name = ('train', 'dev', 'holdout')[selected]
            current[selected] += len(family)
            counts[name][label] += len(family)
            for row in family:
                assignment[row['image_id']] = name
    rows = [{'image_id': r['image_id'], 'relative_path': r['relative_path'], 'label': int(r['label']),
             'group_id': 'family-'+find(r['image_id']), 'split': assignment[r['image_id']]} for r in audit]
    write_csv(output, rows, ['image_id', 'relative_path', 'label', 'group_id', 'split'])
    digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    Path(output).with_suffix('.sha256').write_text(digest+'  '+Path(output).name+'\n', encoding='utf-8')
    summary = {'seed': seed, 'target_fractions': [0.70, 0.15, 0.15], 'counts_by_label': counts,
               'images': len(rows), 'groups': len(groups), 'duplicate_groups': sum(len(g)>1 for g in groups.values()),
               'split_sha256': digest, 'group_overlap': False}
    Path(output).with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return rows


def compute_statistics(root, manifest, output, *, all_labeled=False):
    root, manifest = Path(root), Path(manifest)
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = rows if all_labeled else [r for r in rows if r['split'] == 'train']
    if not selected:
        raise ValueError('no training images for normalization')
    sums, squares = np.zeros(3), np.zeros(3)
    for index, row in enumerate(selected):
        with Image.open(root/row['relative_path']) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)
            pixels = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)/255
        sums += pixels.mean(axis=0)
        squares += (pixels*pixels).mean(axis=0)
        if (index+1) % 1000 == 0:
            print(f'normalization {index+1}/{len(selected)} train images', flush=True)
    mean = sums/len(selected)
    result = {'mean': mean.tolist(), 'std': np.sqrt(squares/len(selected)-mean*mean).tolist(),
              'image_count': len(selected), 'method': 'equal-image RGB moments after EXIF/RGB and bilinear 128x128 resize',
              'split_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'all_labeled': all_labeled}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'split', 'stats', 'refit'])
    parser.add_argument('--root', default='data/official')
    parser.add_argument('--labels', default='data/raw/train_labels.csv')
    parser.add_argument('--audit-dir', default='data/audit')
    parser.add_argument('--manifest', default='data/splits.csv')
    parser.add_argument('--statistics', default='data/statistics.json')
    parser.add_argument('--confirmed-pairs')
    parser.add_argument('--evaluation-freeze', help='required before preparing all-label refit data')
    args = parser.parse_args()
    if args.stage == 'audit':
        audit_images(args.root, args.labels, args.audit_dir)
    elif args.stage == 'split':
        freeze_split(Path(args.audit_dir)/'images.csv', args.manifest, confirmed_pairs=args.confirmed_pairs)
    elif args.stage == 'stats':
        compute_statistics(args.root, args.manifest, args.statistics)
    else:
        if not args.evaluation_freeze or not Path(args.evaluation_freeze).is_file():
            raise ValueError('refit preparation requires the delivered evaluation freeze file')
        source = Path(args.manifest)
        with source.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row['split'] = 'train'
        target = source.with_name('all_labeled.csv')
        write_csv(target, rows, ['image_id', 'relative_path', 'label', 'group_id', 'split'])
        stats = compute_statistics(args.root, target, target.with_name('all_labeled_statistics.json'), all_labeled=True)
        stats['evaluation_freeze_sha256'] = hashlib.sha256(Path(args.evaluation_freeze).read_bytes()).hexdigest()
        target.with_name('all_labeled_statistics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
