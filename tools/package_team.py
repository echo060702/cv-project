"""只打包协作所需文件，生成逐文件SHA256，不复制浏览器或账号状态。"""
import argparse
import hashlib
from pathlib import Path
import zipfile


PACKAGE_ROOT = 'ai_detection_team_20261122'
ALLOWED_DIRECTORIES = ('cvdetect', 'configs', 'tests', 'tools', 'docs', 'validation', '浏览器核验_20261121')
ALLOWED_SUFFIXES = {'.py', '.json', '.jsonl', '.md', '.csv', '.txt', '.pt', '.svg', '.mmd', '.ps1'}
DATA_FILES = ('splits.csv', 'splits.json', 'splits.sha256', 'statistics.json')


def payload_files(root):
    root = Path(root)
    files = [p for p in root.iterdir() if p.is_file() and (p.suffix == '.md' or p.name in ('requirements.txt', '协作任务板.csv', 'tasks.json', '.gitignore', 'team_data.py'))]
    for directory in ALLOWED_DIRECTORIES:
        base = root/directory
        if base.is_dir():
            files.extend(p for p in base.rglob('*') if p.is_file() and p.suffix in ALLOWED_SUFFIXES
                         and '__pycache__' not in p.parts and 'history_smoke' not in p.parts)
    files.extend(root/'data'/name for name in DATA_FILES if (root/'data'/name).is_file())
    for directory in ('audit', 'raw'):
        base = root/'data'/directory
        if base.is_dir():
            files.extend(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.txt'))
    for subset in ('train', 'test'):
        base = root/'data/official/dataset'/subset
        if base.is_dir():
            files.extend(p for p in base.rglob('*') if p.is_file())
    for directory in ('runs/baselines', 'results/baselines', 'results/handoff'):
        base = root/directory
        if base.is_dir():
            files.extend(p for p in base.rglob('*') if p.is_file()
                         and p.suffix in ('.json', '.jsonl', '.csv', '.txt', '.pt', '.png', '.pdf')
                         and '__pycache__' not in p.parts)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def build_archive(root, archive):
    root, archive = Path(root), Path(archive)
    files = payload_files(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in files:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            manifest.append(hashlib.sha256(content).hexdigest()+'  '+relative)
            compression = zipfile.ZIP_STORED if relative.startswith('data/official/') else zipfile.ZIP_DEFLATED
            output.writestr(PACKAGE_ROOT+'/'+relative, content, compress_type=compression)
        output.writestr(PACKAGE_ROOT+'/MANIFEST.sha256', '\n'.join(manifest)+'\n')
    return len(files)


def verify_manifest(root):
    root = Path(root)
    entries = (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()
    expected = set()
    for line in entries:
        digest, name = line.split('  ', 1)
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('manifest path escapes package root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'SHA256 mismatch: {name}')
        expected.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and '__pycache__' not in p.parts}
    if actual != expected:
        raise ValueError(f'file inventory mismatch: {actual ^ expected}')
    return len(entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='dist/AI图像识别_数据与GA六次基线_完整交接包_20261122.zip')
    parser.add_argument('--verify', help='解压后包根目录，只校验不打包')
    args = parser.parse_args()
    if args.verify:
        print(f'SHA256 verified: {verify_manifest(args.verify)} files')
    else:
        root = Path(__file__).resolve().parents[1]
        count = build_archive(root, Path(args.output))
        print(f'Packaged {count} payload files: {args.output}')


if __name__ == '__main__':
    main()
