import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from team_data import paired_view
from team_eval import classification_metrics, image_views, paired_group_bootstrap


class Member3EvaluationTests(unittest.TestCase):
    def test_incomplete_evaluation_cannot_produce_a_freeze(self):
        from tools import run_member3_evaluation as runner
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'dev'
            with patch.object(runner,'OUTPUT',output):
                with self.assertRaises(FileNotFoundError):
                    runner.finalize({'jobs':[{'key':'missing_run'}]})
            self.assertEqual(list(Path(temporary).iterdir()),[])

    def test_invalid_conditions_are_rejected_before_checkpoint_loading(self):
        from team_eval import evaluate
        with self.assertRaisesRegex(ValueError,'unsupported'):
            evaluate('not-a-checkpoint','unused-output',patches=2)

    def test_tied_auc_and_threshold(self):
        metrics = classification_metrics([0, 1, 0, 1], [.5, .5, .1, .9])
        self.assertEqual(metrics['accuracy'], .75)
        self.assertEqual(metrics['roc_auc'], .875)
        self.assertEqual(metrics['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertIsNone(classification_metrics([0], [.1])['roc_auc'])

    def test_independent_center_matches_training_adapter(self):
        rng = np.random.default_rng(17)
        for shape in ((15, 30, 3), (70, 60, 3)):
            image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
            for mode in ('local', 'global'):
                expected, _ = paired_view(image, size=32, mode=mode, rng=rng, training=False, quality_range=(60, 100))
                actual = image_views(image, size=32, mode=mode, patches=1, condition='clean')[0]
                self.assertTrue(torch.equal(actual, expected))

    def test_jpeg_matches_full_image_before_crop_protocol(self):
        rng = np.random.default_rng(1)
        image = Image.fromarray(rng.integers(0, 256, (65, 79, 3), dtype=np.uint8))
        _, expected = paired_view(image, size=32, mode='local', rng=rng, training=False, quality_range=(60, 100))
        actual = image_views(image, size=32, mode='local', patches=1, condition='jpeg80')[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_cluster_bootstrap_and_pair_validation(self):
        a = [dict(image_id=str(i), group_id='same', label=1, prob_fake=.9) for i in range(3)]
        b = [dict(r, prob_fake=.1) for r in a]
        result = paired_group_bootstrap(a, b, iterations=50)
        self.assertEqual(result['groups'], 1)
        self.assertEqual(result['ci95'], [1., 1.])
        with self.assertRaises(ValueError):
            paired_group_bootstrap(a, b[:-1])

    def test_reject_nonfinite_probabilities(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [float('nan'), .5])
