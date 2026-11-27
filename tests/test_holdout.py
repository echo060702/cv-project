import csv
import json
import tempfile
import unittest
from pathlib import Path

from team_eval import classification_metrics, sha
from tools.run_member3_holdout import read_predictions, validate_inventory


class HoldoutBoundaryTests(unittest.TestCase):
    def test_overlap_is_rejected_even_when_holdout_inventory_matches(self):
        rows = [dict(image_id=str(i), group_id=f'g{i}') for i in range(1200)]
        metadata = {'splits': {
            'holdout': {'image_ids': [r['image_id'] for r in rows], 'group_ids': [r['group_id'] for r in rows]},
            'train': {'image_ids': ['other'], 'group_ids': ['g0']},
            'dev': {'image_ids': [], 'group_ids': []}}}
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_inventory(rows, metadata)

    def test_changed_predictions_cannot_be_reused(self):
        rows = [dict(image_id='one', group_id='group', label=1)]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder/'predictions.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image_id', 'group_id', 'label', 'prob_fake'])
                writer.writeheader(); writer.writerow(dict(**rows[0], prob_fake=.9))
            report = dict(protocol_sha256='frozen', predictions_sha256=sha(path), metrics=classification_metrics([1], [.9]))
            (folder/'metrics.json').write_text(json.dumps(report))
            self.assertEqual(read_predictions(folder, rows, 'frozen'), report)
            path.write_text(path.read_text().replace('0.9', '0.1'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_predictions(folder, rows, 'frozen')
