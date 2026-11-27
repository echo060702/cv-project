"""结果核验微型fixture；不产生正式识别成绩。"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


class BaselineResultTests(unittest.TestCase):
    def test_explicit_authorization_is_used_and_stale_inputs_are_rejected(self):
        from cvdetect.engine import source_fingerprint
        from tools.summarize_baselines import main
        valid = dict(source_sha256=source_fingerprint(),
                     data_adapter_sha256=hashlib.sha256(Path('team_data.py').read_bytes()).hexdigest(),
                     split_sha256=hashlib.sha256(Path('data/splits.csv').read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as temporary:
            auth = Path(temporary)/'member3_freeze.json'
            args = ['summarize', '--authorization', str(auth), '--experiments', 'E', 'F']
            auth.write_text(json.dumps(valid), encoding='utf-8')
            # A valid custom freeze reaches run auditing without consulting the legacy path.
            with patch('sys.argv', args), patch('tools.summarize_baselines.inspect_run',
                                               side_effect=RuntimeError('reached run audit')):
                with self.assertRaisesRegex(RuntimeError, 'reached run audit'):
                    main()
            for key in valid:
                with self.subTest(key=key):
                    auth.write_text(json.dumps({**valid, key: '0'*64}), encoding='utf-8')
                    with patch('sys.argv', args), patch('tools.summarize_baselines.inspect_run') as audit:
                        with self.assertRaisesRegex(ValueError, key):
                            main()
                        audit.assert_not_called()

    def test_member3_queue_passes_its_freeze_to_final_summary(self):
        from tools import run_member3
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(run_member3, 'ROOT', Path(temporary)), \
                 patch.object(run_member3, 'freeze_execution'), \
                 patch.object(run_member3.subprocess, 'run') as run, \
                 patch('sys.argv', ['run_member3', '--device', 'cpu']):
                run_member3.main()
            self.assertEqual(run.call_count, 7)  # six training requests and one summary
            command = run.call_args_list[-1].args[0]
            self.assertEqual(command[command.index('--authorization')+1],
                             'validation/member3_execution_freeze.json')

    def fixture(self, root):
        from cvdetect.engine import source_fingerprint
        config = json.loads(Path('configs/G.json').read_text(encoding='utf-8'))
        config['training']['epochs'] = 3
        metadata = {'data_kind': 'official', 'split_sha256': 'a'*64,
                    'splits': {'dev': {'image_ids': ['a', 'b', 'c', 'd'], 'group_ids': ['a', 'b', 'c', 'd']}}}
        history = [dict(epoch=epoch, experiment='G', seed=17, data_kind='official',
                        train_loss=0.5, train_accuracy=0.75, dev_accuracy=0.75,
                        dev_loss=loss, seconds=2., parameters=10, peak_vram_bytes=100, best=epoch < 3)
                   for epoch, loss in ((1, 0.7), (2, 0.6), (3, 0.6))]
        for name, value in (('config.json', config), ('metadata.json', metadata), ('environment.json', {'fixture': True})):
            (root/name).write_text(json.dumps(value), encoding='utf-8')
        (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history), encoding='utf-8')
        for name, epoch in (('best.pt', 2), ('last.pt', 3)):
            torch.save({'epoch': epoch, 'config': config, 'metadata': metadata,
                        'source_sha256': source_fingerprint(), 'metrics': history[epoch-1]}, root/name)
        with (root/'dev_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id', 'group_id', 'label', 'prob_fake'])
            writer.writeheader()
            writer.writerows(dict(image_id=i, group_id=i, label=y, prob_fake=p)
                             for i, y, p in zip('abcd', [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9]))

    def test_results_auditor_is_delivered(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.summarize_baselines'), 'baseline result audit is missing')

    def test_tied_checkpoint_keeps_earlier_epoch_and_accuracy_is_recomputed(self):
        from tools.summarize_baselines import inspect_run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            result, _ = inspect_run(root, expected_epochs=3)
            self.assertEqual(result['best_epoch'], 2)
            self.assertEqual(result['dev_accuracy'], 0.75)
            self.assertEqual(result['dev_count'], 4)
            self.assertEqual(result['recorded_epoch_seconds_total'], 6.)

    def test_incomplete_or_misaligned_artifacts_are_rejected(self):
        from tools.summarize_baselines import inspect_run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root/'metrics.jsonl'
            complete = path.read_text()
            path.write_text('\n'.join(complete.splitlines()[:2])+'\n')
            with self.assertRaisesRegex(ValueError, 'epoch'):
                inspect_run(root, expected_epochs=3)
            path.write_text(complete)
            prediction = root/'dev_predictions.csv'
            prediction.write_text(prediction.read_text().replace('d,d,1,0.9', 'outside,outside,1,0.9'))
            with self.assertRaisesRegex(ValueError, 'dev|prediction'):
                inspect_run(root, expected_epochs=3)


if __name__ == '__main__':
    unittest.main()
