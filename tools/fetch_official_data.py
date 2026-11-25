"""Restore the exact public image archive used by the frozen data handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


REVISION = 'ab5ac7470f495fff579cc1ebae0962a54cedc40d'
URL = f'https://huggingface.co/datasets/emiliiia/ai-generated-image-detection-benchmark/resolve/{REVISION}/dataset.zip'
SHA256 = '3bf72e622a808fbfcb7505ef87dc54e726039a0d6dc7df327cf615cfdb943b38'
SIZE = 2065749665


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='downloads/dataset.zip')
    parser.add_argument('--root', default='data/official')
    parser.add_argument('--direct', action='store_true', help='ignore environment proxy for this download')
    args = parser.parse_args()
    archive, root = Path(args.archive), Path(args.root).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix('.zip.part')
    if not archive.is_file():
        offset = partial.stat().st_size if partial.exists() else 0
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.direct else urllib.request.build_opener()
        request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-'} if offset else {})
        with opener.open(request, timeout=60) as response:
            if offset and response.status != 206:
                offset = 0
            if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise ValueError('unexpected download range')
            with partial.open('ab' if offset else 'wb') as stream:
                total, reported = offset, offset
                while chunk := response.read(4 * 1024 * 1024):
                    stream.write(chunk)
                    total += len(chunk)
                    if total - reported >= 64 * 1024 * 1024:
                        print(f'downloaded {total}/{SIZE} bytes ({100*total/SIZE:.1f}%)', flush=True)
                        reported = total
        if partial.stat().st_size != SIZE:
            raise ValueError('archive size mismatch; retain partial file for resume')
        partial.replace(archive)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if archive.stat().st_size != SIZE or digest != SHA256:
        raise ValueError('official archive size/SHA256 mismatch; do not extract')
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            destination = (root/entry.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError('archive path escapes destination')
        source.extractall(root)
        inventory = len([entry for entry in source.infolist() if not entry.is_dir()])
    report = dict(passed=True, url=URL, revision=REVISION, bytes=SIZE, sha256=digest,
                  extracted_files=inventory, root=str(root), test_pixels_decoded=False)
    output = Path('validation/member2_data_restore.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
