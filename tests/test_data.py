"""真实文件格式的微型fixture；不代表官方图像与算法成绩。"""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        rng = np.random.default_rng(42)
        for label in (0, 1):
            for index in range(12):
                name = f'{label}_{index}.png'
                rgb = rng.integers(0, 256, (39, 45, 3), dtype=np.uint8)
                if index == 11:
                    rgb = np.array(Image.open(self.root/f'{label}_0.png'))
                Image.fromarray(rgb).save(self.root/name)
                self.records.append({'image_id': name, 'label': label, 'image_path': name})
        self.labels = self.root/'labels.csv'
        with self.labels.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'label', 'image_path'])
            writer.writeheader()
            writer.writerows(self.records)

    def prepare(self):
        from tools.prepare_data import audit_images, freeze_split, compute_statistics
        audit = audit_images(self.root, self.labels, self.root/'audit')
        split = self.root/'splits.csv'
        freeze_split(audit, split, seed=17)
        compute_statistics(self.root, split, self.root/'statistics.json')
        config = json.loads(Path('configs/A.json').read_text(encoding='utf-8'))
        config['data'].update(root=str(self.root), manifest=str(split), statistics=str(self.root/'statistics.json'), image_size=32)
        config['training'].update(batch_size=4, workers=0)
        return audit, config

    def test_real_data_module_is_delivered(self):
        self.assertIsNotNone(importlib.util.find_spec('team_data'), 'real paired data adapter is missing')
        self.assertIsNotNone(importlib.util.find_spec('tools.prepare_data'), 'data audit and split implementation is missing')

    def test_exact_duplicate_groups_stay_together_and_split_is_reproducible(self):
        from tools.prepare_data import freeze_split
        audit, config = self.prepare()
        first = Path(config['data']['manifest']).read_bytes()
        freeze_split(audit, self.root/'second.csv', seed=17)
        self.assertEqual(first, (self.root/'second.csv').read_bytes())
        rows = list(csv.DictReader(first.decode().splitlines()))
        assignments = {row['image_id']: row for row in rows}
        for label in (0, 1):
            self.assertEqual(assignments[f'{label}_0.png']['group_id'], assignments[f'{label}_11.png']['group_id'])
            self.assertEqual(assignments[f'{label}_0.png']['split'], assignments[f'{label}_11.png']['split'])
        self.assertEqual(len(rows), 24)
        self.assertEqual({row['split'] for row in rows}, {'train', 'dev', 'holdout'})
        # A holdout image may be unavailable without being read by stats/loaders.
        holdout = next(row for row in rows if row['split'] == 'holdout')
        (self.root/holdout['relative_path']).unlink()
        from tools.prepare_data import compute_statistics
        from team_data import build_loaders
        before = json.loads(Path(config['data']['statistics']).read_text())
        compute_statistics(self.root, config['data']['manifest'], self.root/'again.json')
        self.assertEqual(before['mean'], json.loads((self.root/'again.json').read_text())['mean'])
        bundle = build_loaders(config, seed=17)
        self.assertNotIn(holdout['image_id'], bundle.train_loader.dataset.image_ids)
        self.assertNotIn(holdout['image_id'], bundle.dev_loader.dataset.image_ids)

    def test_epoch_replay_and_paired_geometry_with_global_and_local_modes(self):
        from team_data import build_loaders
        _, config = self.prepare()
        for mode in ('global', 'local'):
            config['data']['mode'] = mode
            bundle = build_loaders(config, seed=17)
            bundle.set_epoch(2)
            first = list(bundle.train_loader)
            bundle.set_epoch(3)
            changed = list(bundle.train_loader)
            bundle.set_epoch(2)
            replay = list(bundle.train_loader)
            self.assertEqual([i for b in first for i in b['image_id']], [i for b in replay for i in b['image_id']])
            self.assertTrue(all(torch.equal(a['image'], b['image']) and torch.equal(a['image_jpeg'], b['image_jpeg']) for a, b in zip(first, replay)))
            self.assertFalse(torch.equal(first[0]['image'], changed[0]['image']))
            self.assertEqual(first[0]['image'].shape, (4, 3, 32, 32))
            self.assertLess(float((first[0]['image']-first[0]['image_jpeg']).abs().mean()), 0.15)
            self.assertGreaterEqual(float(first[0]['image'].min()), 0)
            self.assertLessEqual(float(first[0]['image'].max()), 1)

    def test_manifest_tamper_and_group_leakage_are_rejected_before_loading(self):
        from team_data import build_loaders
        _, config = self.prepare()
        path = Path(config['data']['manifest'])
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError, 'SHA256|sha256|hash'):
            build_loaders(config, seed=17)

    def test_test_directory_is_rejected_by_audit(self):
        from tools.prepare_data import audit_images
        self.labels.write_text('image_id,label,image_path\nforbidden.png,0,dataset/test/forbidden.png\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'training path'):
            audit_images(self.root, self.labels, self.root/'audit')

    def test_refit_data_preparation_requires_freeze_then_covers_all_labeled(self):
        import subprocess
        import sys
        from team_data import build_refit_loaders
        _, config = self.prepare()
        command = [sys.executable, '-X', 'utf8', '-m', 'tools.prepare_data', 'refit',
                   '--root', str(self.root), '--manifest', config['data']['manifest']]
        missing = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertNotEqual(missing.returncode, 0)
        freeze = self.root/'evaluation_freeze.json'
        freeze.write_text('{"purpose":"test_fixture_only"}', encoding='utf-8')
        done = subprocess.run(command+['--evaluation-freeze', str(freeze)], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(done.returncode, 0, done.stderr)
        config['data'].update(manifest=str(self.root/'all_labeled.csv'), statistics=str(self.root/'all_labeled_statistics.json'))
        bundle = build_refit_loaders(config, seed=17)
        self.assertIsNone(bundle.dev_loader)
        self.assertEqual(len(bundle.train_loader.dataset), 24)
        self.assertEqual(bundle.metadata['splits']['holdout']['image_ids'], [])

    def test_multibatch_cpu_resume_replays_actual_image_pipeline(self):
        import copy
        import contextlib
        import io
        from cvdetect.engine import run_training
        from team_data import build_loaders
        _, config = self.prepare()
        config['status'] = 'frozen_budget'
        config['model']['width'] = 4
        config['training'].update(epochs=2, amp=False)
        torch.set_num_threads(2)
        with contextlib.redirect_stdout(io.StringIO()):
            run_training(copy.deepcopy(config), build_loaders(config, seed=17), self.root/'full', device='cpu')
            run_training(copy.deepcopy(config), build_loaders(config, seed=17), self.root/'resume', device='cpu', run_epochs=1)
            run_training(copy.deepcopy(config), build_loaders(config, seed=17), self.root/'resume', device='cpu', resume=self.root/'resume/last.pt')
        full = torch.load(self.root/'full/last.pt', weights_only=True)
        resumed = torch.load(self.root/'resume/last.pt', weights_only=True)
        for key, value in full['model'].items():
            self.assertTrue(torch.equal(value, resumed['model'][key]), key)

    def test_small_grayscale_and_rgba_inputs_produce_rgb_pairs(self):
        from team_data import paired_view
        for image in (Image.new('L', (8, 12), 128), Image.new('RGBA', (12, 8), (30, 80, 140, 70))):
            for mode in ('local', 'global'):
                normal, jpeg = paired_view(image, size=32, mode=mode, rng=np.random.default_rng(4), training=False, quality_range=(75, 75))
                self.assertEqual(normal.shape, (3, 32, 32))
                self.assertEqual(jpeg.shape, normal.shape)
                self.assertTrue(torch.isfinite(normal).all())


if __name__ == '__main__':
    unittest.main()
