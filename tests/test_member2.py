"""Check the boundary that prevents an expensive run with changed recipes."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools.run_member2 import ROOT, freeze_execution


class ExecutionFreezeTests(unittest.TestCase):
    def test_pause_marker_prevents_any_training_launch(self):
        import tools.run_member2 as runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/'runs/member2'
            output.mkdir(parents=True)
            (output/'PAUSE_AFTER_CURRENT').write_text('pause')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'freeze_execution'), \
                 patch.object(runner.subprocess, 'run') as launch, patch('sys.argv', ['run_member2']):
                runner.main()
                launch.assert_not_called()
            self.assertTrue((output/'PAUSE_AFTER_CURRENT').exists())
            self.assertFalse((output/'.runner.lock').exists())

    def fixture(self, directory):
        for name in ('configs/B.json', 'configs/C.json', 'configs/D.json',
                     'runs/baselines/A_seed17/config.json', 'runs/baselines/A_seed17/metadata.json',
                     'results/baselines/baseline_summary.json', 'team_data.py', 'tools/prepare_data.py',
                     'data/splits.csv', 'data/statistics.json'):
            target = directory/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        (directory/'validation').mkdir()
        (directory/'validation/member2_data_pipeline.json').write_text(json.dumps(dict(passed=True, statistics_recomputed_train_only=True)))

    def test_changed_budget_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root/'configs/B.json'
            config = json.loads(path.read_text())
            config['training']['epochs'] = 21
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'common budget'):
                freeze_execution(root)
            self.assertFalse((root/'validation/member2_execution_freeze.json').exists())

    def test_freeze_reuse_and_data_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            first = freeze_execution(root)
            self.assertEqual(first, freeze_execution(root))
            with (root/'data/splits.csv').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'split/statistics'):
                freeze_execution(root)


if __name__ == '__main__':
    unittest.main()
