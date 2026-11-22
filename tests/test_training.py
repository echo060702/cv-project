import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


def fixture_config(experiment='D'):
    branches = {'B': 'residual', 'C': 'none', 'D': 'residual'}
    return {'experiment': experiment, 'status': 'engineering_test', 'seed': 17,
            'model': {'width': 4, 'branch': branches[experiment]},
            'data': {'mode': 'local', 'image_size': 32, 'root': ''},
            'training': {'epochs': 2, 'batch_size': 8, 'learning_rate': 0.003,
                         'weight_decay': 0.0001, 'consistency': 0. if experiment == 'B' else 0.1,
                         'warmup_fraction': 0.1, 'workers': 0, 'amp': False}}


class TrainingTests(unittest.TestCase):
    def test_official_training_requires_frozen_shared_budget(self):
        from cvdetect.engine import run_training
        from tests.fixtures.synthetic_adapter import build_loaders
        config = fixture_config()
        bundle = build_loaders(config, seed=17)
        bundle.metadata['data_kind'] = 'official'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'frozen_budget'):
                run_training(config, bundle, folder, device='cpu')
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_invalid_raw_input_is_rejected_at_boundary(self):
        from cvdetect.contracts import validate_bundle
        from tests.fixtures.synthetic_adapter import build_loaders
        config = fixture_config()
        bundle = build_loaders(config, seed=17)
        for record in bundle.dev_loader.dataset.records:
            record['image'] = record['image'] * 0 + 2
        with self.assertRaisesRegex(ValueError, r'\[0,1\]'):
            validate_bundle(bundle, config)

    def test_final_refit_has_no_dev_selection_and_keeps_separate_provenance(self):
        self.assertIsNotNone(importlib.util.find_spec('cvdetect.refit'), '最终全量训练入口尚未实现')
        from cvdetect.refit import run_refit
        from tests.fixtures.synthetic_adapter import build_loaders
        config = fixture_config('B')
        bundle = build_loaders(config, seed=17)
        from torch.utils.data import ConcatDataset, DataLoader
        loader = DataLoader(ConcatDataset([bundle.train_loader.dataset, bundle.dev_loader.dataset]), batch_size=8)
        for key in ('image_ids', 'group_ids'):
            bundle.metadata['splits']['train'][key].extend(bundle.metadata['splits']['dev'][key])
            bundle.metadata['splits']['dev'][key] = []
        recipe = {'epochs': 2, 'selected_experiment': 'B', 'selection_summary_sha256': 'b'*64,
                  'evaluation_freeze_sha256': 'c'*64}
        with tempfile.TemporaryDirectory() as temp:
            run_refit(config, loader, bundle.metadata, bundle.set_epoch, recipe, temp, device='cpu')
            ckpt = torch.load(Path(temp)/'final.pt', weights_only=True)
            self.assertEqual(ckpt['purpose'], 'final_refit_no_validation')
            self.assertNotIn('dev_accuracy', ckpt['metrics'])
            self.assertEqual(ckpt['recipe'], recipe)

    def test_epoch_resume_matches_uninterrupted_training_and_loads_predictions(self):
        self.assertIsNotNone(importlib.util.find_spec('cvdetect.engine'), '训练功能尚未实现')
        from cvdetect.engine import run_training
        from cvdetect.inference import load_detector
        from tests.fixtures.synthetic_adapter import build_loaders
        torch.set_num_threads(2)
        config = fixture_config()
        with tempfile.TemporaryDirectory() as temp:
            whole, resumed = Path(temp)/'whole', Path(temp)/'resumed'
            run_training(config, build_loaders(config, seed=17), whole, device='cpu')
            run_training(config, build_loaders(config, seed=17), resumed, device='cpu', run_epochs=1)
            run_training(config, build_loaders(config, seed=17), resumed, device='cpu', resume=resumed/'last.pt')
            a = torch.load(whole/'last.pt', weights_only=True)
            b = torch.load(resumed/'last.pt', weights_only=True)
            self.assertEqual(a['epoch'], 2)
            self.assertTrue(all(torch.equal(a['model'][key], b['model'][key]) for key in a['model']))
            self.assertEqual(a['optimizer']['param_groups'][0]['lr'], b['optimizer']['param_groups'][0]['lr'])
            history = [json.loads(line) for line in (resumed/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in history], [1, 2])
            self.assertEqual(history[0]['data_kind'], 'synthetic_engineering')
            model, checkpoint = load_detector(resumed/'best.pt', device='cpu')
            batch = next(iter(build_loaders(config, seed=17).dev_loader))
            with torch.no_grad():
                probabilities = model(batch['image']).sigmoid()
            with (resumed/'dev_predictions.csv').open(encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r['image_id'].startswith('synthetic-dev') for r in rows))
            self.assertTrue(torch.allclose(probabilities, torch.tensor([float(r['prob_fake']) for r in rows[:8]]), atol=1e-6))
            different = build_loaders(config, seed=17)
            different.metadata['data_revision'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'metadata'):
                run_training(config, different, resumed, device='cpu', resume=resumed/'last.pt')


if __name__ == '__main__':
    unittest.main()
