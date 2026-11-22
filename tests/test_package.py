import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


class PackageTests(unittest.TestCase):
    def test_allowlist_and_hashes_preserve_team_files_without_private_state(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.package_team'), '协作打包工具尚未实现')
        from tools.package_team import build_archive, verify_manifest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'source'
            root.mkdir()
            (root/'README.md').write_text('入口', encoding='utf-8')
            (root/'requirements.txt').write_text('torch', encoding='utf-8')
            (root/'cvdetect').mkdir()
            (root/'cvdetect'/'__init__.py').write_text('', encoding='utf-8')
            (root/'docs').mkdir()
            (root/'docs'/'分工.md').write_text('协作', encoding='utf-8')
            (root/'.desktop-read-session').mkdir()
            (root/'.desktop-read-session'/'secret.json').write_text('private', encoding='utf-8')
            (root/'data').mkdir()
            (root/'data'/'test_labels.csv').write_text('do not package', encoding='utf-8')
            (root/'team_data.py').write_text('# data adapter', encoding='utf-8')
            (root/'data'/'splits.csv').write_text('image_id,label,split', encoding='utf-8')
            (root/'data'/'statistics.json').write_text('{}', encoding='utf-8')
            training = root/'data'/'official'/'dataset'/'train'
            training.mkdir(parents=True)
            (training/'train.png').write_bytes(b'training fixture')
            test = training.parent/'test'
            test.mkdir()
            (test/'image_0001.jpg').write_bytes(b'unlabeled test fixture')
            (root/'data'/'dataset.zip').write_bytes(b'do not package redundant source archive')
            baselines = root/'runs'/'baselines'/'G_seed17'
            baselines.mkdir(parents=True)
            (baselines/'best.pt').write_bytes(b'official checkpoint fixture')
            figures = root/'results'/'baselines'
            figures.mkdir(parents=True)
            (figures/'training_curves.pdf').write_bytes(b'plot fixture')
            archive = Path(temp)/'team.zip'
            build_archive(root, archive)
            target = Path(temp)/'unpacked'
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertFalse(any('secret' in name or 'test_labels' in name for name in z.namelist()))
                names = z.namelist()
                self.assertTrue(any(name.endswith('/team_data.py') for name in names), 'missing real adapter')
                self.assertTrue(any(name.endswith('/data/splits.csv') for name in names), 'missing frozen split')
                self.assertTrue(any(name.endswith('/data/statistics.json') for name in names), 'missing train statistics')
                self.assertTrue(any(name.endswith('/dataset/train/train.png') for name in names), 'missing training data')
                self.assertTrue(any(name.endswith('/dataset/test/image_0001.jpg') for name in names), 'missing isolated unlabeled test data')
                self.assertTrue(any(name.endswith('/runs/baselines/G_seed17/best.pt') for name in names), 'missing official baseline checkpoint')
                self.assertTrue(any(name.endswith('/results/baselines/training_curves.pdf') for name in names), 'missing baseline plot')
                self.assertFalse(any(name.endswith('/dataset.zip') for name in names))
                z.extractall(target)
            package = next(target.iterdir())
            self.assertGreaterEqual(verify_manifest(package), 4)
            (package/'README.md').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_manifest(package)


if __name__ == '__main__':
    unittest.main()
