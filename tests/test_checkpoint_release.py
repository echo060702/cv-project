import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.pack_checkpoints import collect, build


class CheckpointReleaseTests(unittest.TestCase):
    def test_member3_paths_and_last_checkpoint_survive_packaging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/'runs/member3/E_seed17'
            run.mkdir(parents=True)
            (run/'best.pt').write_bytes(b'best fixture')
            (run/'last.pt').write_bytes(b'last fixture')
            with patch('tools.pack_checkpoints.Path.cwd', return_value=root):
                items = collect(root/'runs/member3', ['E'], [17], include_last=True)
            report = build(items, root/'release.zip')
            with zipfile.ZipFile(report['output']) as archive:
                self.assertEqual(set(archive.namelist()), {'runs/member3/E_seed17/best.pt', 'runs/member3/E_seed17/last.pt'})
                self.assertEqual(archive.read('runs/member3/E_seed17/last.pt'), b'last fixture')
            (run/'last.pt').unlink()
            with patch('tools.pack_checkpoints.Path.cwd', return_value=root):
                with self.assertRaises(SystemExit):
                    collect(root/'runs/member3', ['E'], [17], include_last=True)
                self.assertEqual(len(collect(root/'runs/member3', ['E'], [17])), 1)
