import json
from pathlib import Path
import unittest

from cvdetect.contracts import validate_config
from cvdetect.model import Detector


class ConfigTests(unittest.TestCase):
    def test_seven_configs_share_recipe_and_match_factorial_design(self):
        root = Path(__file__).resolve().parents[1]
        files = sorted((root/'configs').glob('*.json'))
        self.assertEqual(len(files), 7, '七组配置尚未交付')
        loaded = [json.loads(p.read_text(encoding='utf-8')) for p in files]
        recipes = []
        for config in loaded:
            validate_config(config)
            train = dict(config['training'])
            train.pop('consistency')
            recipes.append(train)
        self.assertTrue(all(r == recipes[0] for r in recipes))
        branches = {c['experiment']: c['model']['branch'] for c in loaded}
        self.assertEqual(branches, {'G': 'none', 'A': 'none', 'B': 'residual', 'C': 'none', 'D': 'residual', 'E': 'rgb', 'F': 'rgb'})


if __name__ == '__main__':
    unittest.main()
