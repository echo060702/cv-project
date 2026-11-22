import importlib.util
import unittest

import torch


class ModelTests(unittest.TestCase):
    def test_outputs_gradients_and_capacity_controls(self):
        self.assertIsNotNone(importlib.util.find_spec('cvdetect.model'), '模型功能尚未实现')
        from cvdetect.model import Detector, highpass
        torch.set_num_threads(2)
        x = torch.rand(4, 3, 32, 32)
        counts = {}
        for branch in ('none', 'residual', 'rgb'):
            model = Detector(width=4, branch=branch)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (4,))
            logits.square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            counts[branch] = sum(p.numel() for p in model.parameters())
        self.assertEqual(counts['residual'], counts['rgb'])
        self.assertGreater(counts['residual'], counts['none'])
        self.assertTrue(torch.allclose(highpass(torch.ones_like(x)), torch.zeros_like(x), atol=1e-6))

    def test_seeded_initialization_is_reproducible(self):
        self.assertIsNotNone(importlib.util.find_spec('cvdetect.model'), '模型功能尚未实现')
        from cvdetect.model import Detector
        torch.manual_seed(17)
        first = Detector(width=4, branch='residual')
        torch.manual_seed(17)
        second = Detector(width=4, branch='residual')
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())))


class LossTests(unittest.TestCase):
    def test_js_is_symmetric_stable_and_zero_for_identical_predictions(self):
        self.assertIsNotNone(importlib.util.find_spec('cvdetect.losses'), '损失功能尚未实现')
        from cvdetect.losses import paired_loss, js_divergence
        a = torch.tensor([-1000., 1000., 0.2], requires_grad=True)
        b = torch.tensor([1000., -1000., -0.3], requires_grad=True)
        js = js_divergence(a, b)
        self.assertTrue(torch.isfinite(js))
        self.assertGreaterEqual(js.item(), 0)
        self.assertLessEqual(js.item(), 0.694)
        self.assertTrue(torch.allclose(js, js_divergence(b, a)))
        self.assertAlmostEqual(js_divergence(a, a).item(), 0., places=6)
        loss, parts = paired_loss(a, b, torch.tensor([0., 1., 1.]), 0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())
        supervised, zero_parts = paired_loss(a, b, torch.tensor([0., 1., 1.]), 0.)
        expected = 0.5 * (torch.nn.functional.binary_cross_entropy_with_logits(a, torch.tensor([0., 1., 1.])) + torch.nn.functional.binary_cross_entropy_with_logits(b, torch.tensor([0., 1., 1.])))
        self.assertTrue(torch.allclose(supervised, expected))


class ContractTests(unittest.TestCase):
    def test_group_leakage_and_test_split_are_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('cvdetect.contracts'), '数据边界尚未实现')
        from cvdetect.contracts import validate_metadata
        meta = {
            'data_kind': 'synthetic_engineering', 'data_revision': 'fixture-v1',
            'split_sha256': 'a' * 64, 'mean': [0.5]*3, 'std': [0.5]*3,
            'splits': {
                'train': {'image_ids': ['a'], 'group_ids': ['g1']},
                'dev': {'image_ids': ['b'], 'group_ids': ['g2']},
                'holdout': {'image_ids': ['c'], 'group_ids': ['g3']},
            },
        }
        validate_metadata(meta)
        meta['splits']['dev']['group_ids'] = ['g1']
        with self.assertRaisesRegex(ValueError, 'group'):
            validate_metadata(meta)
        meta['splits']['dev']['group_ids'] = ['g2']
        meta['splits']['test'] = {'image_ids': [], 'group_ids': []}
        with self.assertRaisesRegex(ValueError, 'split'):
            validate_metadata(meta)


if __name__ == '__main__':
    unittest.main()
