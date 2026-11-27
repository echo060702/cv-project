"""打包一组 checkpoint 为 Release 附件，并给出 SHA256 与发布说明片段。

用法：
    python -X utf8 -m tools.pack_checkpoints --experiments B C D --seeds 17 29 43 --output dist/bcd_checkpoints.zip

只收集 <runs>/<实验>_seed<种子>/best.pt，缺失时直接报错（避免发出不完整的权重包）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from cvdetect.contracts import validate_config  # noqa: F401  (保持与训练代码同源的可导入性)


def collect(runs_dir: Path, experiments: list[str], seeds: list[int], *, include_last: bool = False) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    missing: list[str] = []
    for exp in experiments:
        for seed in seeds:
            run = runs_dir / f'{exp}_seed{seed}'
            for filename in (('best.pt', 'last.pt') if include_last else ('best.pt',)):
                weight = run / filename
                if weight.is_file():
                    relative = weight.resolve().relative_to(Path.cwd().resolve()).as_posix()
                    items.append((relative, weight))
                else:
                    missing.append(str(weight))
    if missing:
        raise SystemExit('缺少 checkpoint：\n  ' + '\n  '.join(missing))
    return items


def build(items: list[tuple[str, Path]], output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in items:
            archive.write(path, relative)
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    sidecar = output.with_suffix(output.suffix + '.sha256')
    sidecar.write_text(f'{digest}  {output.name}\n', encoding='utf-8')
    return {'output': str(output), 'bytes': output.stat().st_size, 'sha256': digest,
            'sidecar': str(sidecar), 'files': [relative for relative, _ in items]}


def release_body(items: list[tuple[str, Path]], report: dict, experiments: list[str], seeds: list[int]) -> str:
    rows = '\n'.join(f'| `{relative}` | {path.stat().st_size / 1024 ** 2:.1f} MB |' for relative, path in items)
    return f'''## Checkpoint 权重包

| 文件 | 大小 |
|---|---|
{rows}

- 实验：{', '.join(experiments)}；种子：{', '.join(str(s) for s in seeds)}
- 大小：{report['bytes']:,} 字节（{report['bytes'] / 1024 ** 2:.2f} MiB），{len(items)} 个文件
- SHA256：`{report['sha256']}`

### 用法

```bash
git clone https://github.com/echo060702/cv-project.git && cd cv-project
curl -L -o {Path(report['output']).name} <本页附件下载链接>
sha256sum {Path(report['output']).name}   # 必须等于上面的 SHA256
unzip -o {Path(report['output']).name} -d .
```

附件保留训练器保存的原始checkpoint（含优化器状态）；是否包含last.pt以文件表为准。
'''


def main() -> None:
    parser = argparse.ArgumentParser(description='打包 checkpoint 为 Release 附件并输出 SHA256')
    parser.add_argument('--runs', default='runs/baselines')
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-last', action='store_true', help='also include last.pt for complete run auditing')
    parser.add_argument('--release-body', help='把发布说明片段写入该文件')
    args = parser.parse_args()

    items = collect(Path(args.runs), args.experiments, args.seeds, include_last=args.include_last)
    report = build(items, Path(args.output))
    if args.release_body:
        Path(args.release_body).write_text(release_body(items, report, args.experiments, args.seeds), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
