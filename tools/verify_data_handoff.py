"""核验已交付真实数据，不训练模型，不解码官方test像素。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from team_data import build_loaders
from tools.prepare_data import compute_statistics


def verify_data(*, recompute_statistics=False):
    def rows(path):
        with Path(path).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    labels = rows('data/raw/train_labels.csv')
    audit = rows('data/audit/images.csv')
    split = rows('data/splits.csv')
    digest = hashlib.sha256(Path('data/splits.csv').read_bytes()).hexdigest()
    summary = json.loads(Path('data/splits.json').read_text(encoding='utf-8'))
    stats = json.loads(Path('data/statistics.json').read_text(encoding='utf-8'))
    assert digest == summary['split_sha256'] == stats['split_sha256']
    assert Path('data/splits.sha256').read_text().split()[0] == digest
    assert len(labels) == len(audit) == len(split) == 8000
    official = {r['image_id']: (r['label'], r['image_path']) for r in labels}
    assert len(official) == 8000
    assert {r['image_id'] for r in audit} == {r['image_id'] for r in split} == set(official)
    root = Path('data/official').resolve()
    for r in audit:
        path = (root/r['relative_path']).resolve()
        assert path.is_relative_to(root/'dataset/train')
        assert (r['label'], r['relative_path']) == official[r['image_id']]
        assert r['status'] == 'ok'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r['file_sha256'], r['image_id']
    actual = {p.relative_to(root).as_posix() for p in (root/'dataset/train').rglob('*') if p.is_file()}
    assert actual == {r['relative_path'] for r in audit}
    groups = {}
    counts = Counter()
    for r in split:
        assert (r['label'], r['relative_path']) == official[r['image_id']]
        groups.setdefault(r['group_id'], set()).add(r['split'])
        counts[r['split'], r['label']] += 1
    assert all(len(s) == 1 for s in groups.values())
    assert counts == Counter({('train', '0'): 2800, ('train', '1'): 2800,
                              ('dev', '0'): 600, ('dev', '1'): 600,
                              ('holdout', '0'): 600, ('holdout', '1'): 600})
    assert stats['image_count'] == 5600 and stats['all_labeled'] is False
    if recompute_statistics:
        with tempfile.TemporaryDirectory() as temporary:
            computed = compute_statistics(root, 'data/splits.csv', Path(temporary)/'stats.json')
        for key in ('mean', 'std'):
            np.testing.assert_allclose(computed[key], stats[key], rtol=0, atol=1e-12)

    batch_checks = []
    for exp in ('G', 'A'):
        config = json.loads(Path(f'configs/{exp}.json').read_text(encoding='utf-8'))
        bundle = build_loaders(config, seed=17)
        assert len(bundle.train_loader.dataset) == 5600 and len(bundle.dev_loader.dataset) == 1200
        bundle.set_epoch(2)
        first = next(iter(bundle.train_loader))
        bundle.set_epoch(2)
        replay = next(iter(bundle.train_loader))
        assert first['image_id'] == replay['image_id']
        for key in ('image', 'image_jpeg', 'label'):
            assert torch.equal(first[key], replay[key])
        for name, batch in (('train', first), ('dev', next(iter(bundle.dev_loader)))):
            assert tuple(batch['image'].shape) == tuple(batch['image_jpeg'].shape) == (16, 3, 224, 224)
            for key in ('image', 'image_jpeg'):
                assert torch.isfinite(batch[key]).all() and batch[key].min() >= 0 and batch[key].max() <= 1
            assert all(str(int(label)) == official[identity][0] for identity, label in zip(batch['image_id'], batch['label']))
            assert set(batch['image_id']) <= set(bundle.metadata['splits'][name]['image_ids'])
        batch_checks.append({'experiment': exp, 'mode': config['data']['mode'], 'shape': [16, 3, 224, 224], 'epoch_replay_equal': True})

    tests = rows('data/audit/test_files.csv')
    assert len(tests) == 1000
    test_names = {p.name for p in (root/'dataset/test').iterdir() if p.is_file()}
    assert test_names == {r['image_id'] for r in tests}
    for r in tests:
        content = (root/'dataset/test'/r['image_id']).read_bytes()
        assert len(content) == int(r['bytes']) and hashlib.sha256(content).hexdigest() == r['file_sha256']
    template = rows('data/raw/sample_submission.csv')
    assert len(template) == 1000 and set(template[0]) == {'image_id', 'label'}
    assert len({r['image_id'] for r in template}) == 1000 and {r['image_id'] for r in template} == test_names
    assert all(r['label'] == '0' for r in template)
    return {'passed': True, 'training_files_sha256_verified': 8000, 'test_files_byte_hash_verified': 1000,
            'split_sha256': digest, 'counts_by_split_label': {f'{s}/{l}': n for (s, l), n in sorted(counts.items())},
            'statistics_recomputed_train_only': recompute_statistics, 'batch_checks': batch_checks,
            'test_pixels_decoded': False, 'holdout_loader_created': False, 'model_training_performed': False,
            'submission_template_ids_verified': 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/received_data_check.json')
    parser.add_argument('--recompute-statistics', action='store_true')
    args = parser.parse_args()
    result = verify_data(recompute_statistics=args.recompute_statistics)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
