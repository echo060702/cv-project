"""Package complete member2 evidence and checkpoints, without images or environments."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from tools.package_team import payload_files
from tools.run_member2 import EXPERIMENTS, ROOT, SEEDS


def verify_archive(archive):
    with zipfile.ZipFile(archive) as source:
        entries = source.read('cv-project-member2/MANIFEST.sha256').decode('utf-8').splitlines()
        expected = {}
        for line in entries:
            digest, relative = line.split('  ', 1)
            if relative.startswith('/') or '..' in Path(relative).parts:
                raise ValueError('invalid manifest path')
            expected['cv-project-member2/'+relative] = digest
        actual = set(source.namelist())-{'cv-project-member2/MANIFEST.sha256'}
        if actual != set(expected):
            raise ValueError('archive inventory differs from manifest')
        for name, digest in expected.items():
            with source.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'archive SHA256 mismatch: {name}')
        completed = 0
        for seed in SEEDS:
            for experiment in EXPERIMENTS:
                prefix = f'cv-project-member2/runs/member2/{experiment}_seed{seed}/'
                history = [json.loads(line) for line in source.read(prefix+'metrics.jsonl').decode('utf-8').splitlines()]
                if [r['epoch'] for r in history] != list(range(1, 21)):
                    raise ValueError('incomplete run inside archive')
                for name in ('best.pt', 'last.pt', 'config.json', 'metadata.json', 'environment.json', 'dev_predictions.csv'):
                    if prefix+name not in expected:
                        raise ValueError(f'missing run artifact: {prefix+name}')
                completed += 1
    return dict(passed=True, payload_files=len(expected), complete_member2_runs=completed,
                all_file_hashes_verified=True, images_included=False, environment_included=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/member2_BCD_delivery.zip')
    parser.add_argument('--verify', help='verify an existing ZIP only')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_archive(args.verify), indent=2))
        return
    report = json.loads((ROOT/'results/member2/ablation_summary.json').read_text(encoding='utf-8'))
    if not report['passed'] or report['completed_runs'] != 9 or not report['checkpoint_predictions_replayed']:
        raise ValueError('complete and replay all nine official runs before packaging')
    paths = [p for p in payload_files(ROOT) if not p.relative_to(ROOT).as_posix().startswith('data/official/')]
    archive = Path(args.output)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise ValueError('archive exists; choose a new version instead of overwriting')
    manifest = []
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            manifest.append(f'{digest}  {relative}')
            target.write(path, 'cv-project-member2/'+relative)
        target.writestr('cv-project-member2/MANIFEST.sha256', '\n'.join(manifest)+'\n')
    verification = verify_archive(archive)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    verification.update(archive=str(archive), bytes=archive.stat().st_size, sha256=digest)
    archive.with_suffix('.verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
