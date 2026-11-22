"""成员1正式数据适配器：只加载冻结清单中的train/dev。"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version
import torch
from torch.utils.data import DataLoader, Dataset

from cvdetect.contracts import DataBundle, validate_metadata


def paired_view(image, *, size, mode, rng, training, quality_range):
    image = ImageOps.exif_transpose(image).convert('RGB')
    if training and rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if mode == 'global':
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    else:
        rgb = np.asarray(image)
        pad_y, pad_x = max(0, size-rgb.shape[0]), max(0, size-rgb.shape[1])
        if pad_y or pad_x:
            rgb = np.pad(rgb, ((pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2), (0, 0)), mode='reflect')
            image = Image.fromarray(rgb)
    quality = int(rng.integers(quality_range[0], quality_range[1]+1)) if training else 80
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, subsampling=0)
    with Image.open(io.BytesIO(buffer.getvalue())) as encoded:
        jpeg = encoded.convert('RGB')
    width, height = image.size
    if training:
        x = int(rng.integers(0, width-size+1))
        y = int(rng.integers(0, height-size+1))
    else:
        x, y = (width-size)//2, (height-size)//2
    box = (x, y, x+size, y+size)
    normal = torch.from_numpy(np.array(image.crop(box), copy=True)).permute(2, 0, 1).float()/255
    encoded = torch.from_numpy(np.array(jpeg.crop(box), copy=True)).permute(2, 0, 1).float()/255
    return normal, encoded


class ImagePairs(Dataset):
    def __init__(self, root, rows, config, seed, training):
        self.root, self.rows, self.config = root, rows, config
        self.seed, self.training, self.epoch = seed, training, 0
        self.image_ids = [r['image_id'] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        identity = f'{self.seed}:{self.epoch if self.training else 0}:{row["image_id"]}'
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little'))
        with Image.open(self.root/row['relative_path']) as source:
            normal, jpeg = paired_view(source, size=self.config['image_size'], mode=self.config['mode'], rng=rng,
                                       training=self.training, quality_range=self.config['jpeg_quality'])
        return {'image': normal, 'image_jpeg': jpeg, 'label': float(row['label']),
                'image_id': row['image_id'], 'group_id': row['group_id']}


def _build(config, seed, refit):
    data = config['data']
    root, manifest = Path(data['root']).resolve(), Path(data['manifest'])
    default_stats = 'all_labeled_statistics.json' if refit else 'statistics.json'
    stats_path = Path(data.get('statistics', manifest.with_name(default_stats)))
    statistics = json.loads(stats_path.read_text(encoding='utf-8'))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if statistics['split_sha256'] != digest:
        raise ValueError('statistics manifest SHA256 mismatch')
    if bool(statistics['all_labeled']) != refit:
        raise ValueError('normalization scope differs from research/refit purpose')
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if data['mode'] not in ('global', 'local') or not 1 <= data['jpeg_quality'][0] <= data['jpeg_quality'][1] <= 100:
        raise ValueError('invalid global/local or JPEG quality settings')
    if config['training']['workers'] != 0:
        raise ValueError('this frozen adapter uses workers=0; parallel loader requires separately verified revision')
    for row in rows:
        path = (root/row['relative_path']).resolve()
        if not path.is_relative_to(root) or 'test' in Path(row['relative_path']).parts or row['label'] not in ('0', '1'):
            raise ValueError('manifest contains invalid training path/label')
    split_rows = {name: [r for r in rows if r['split'] == name] for name in ('train', 'dev', 'holdout')}
    if sum(map(len, split_rows.values())) != len(rows):
        raise ValueError('manifest contains unknown split')
    revision = hashlib.sha256(Path(__file__).read_bytes()+Path('tools/prepare_data.py').read_bytes()+stats_path.read_bytes()).hexdigest()
    metadata = {'data_kind': 'official', 'data_revision': 'paired-rgb-v1-'+revision,
                'split_sha256': digest, 'mean': statistics['mean'], 'std': statistics['std'],
                'pillow_version': pillow_version, 'statistics_sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                'splits': {name: {'image_ids': [r['image_id'] for r in members], 'group_ids': [r['group_id'] for r in members]}
                           for name, members in split_rows.items()}}
    validate_metadata(metadata, refit=refit)
    train = ImagePairs(root, split_rows['train'], data, seed, True)
    generator = torch.Generator()
    train_loader = DataLoader(train, batch_size=config['training']['batch_size'], shuffle=True, generator=generator, num_workers=0)
    dev_loader = None
    if not refit:
        dev = ImagePairs(root, split_rows['dev'], data, seed, False)
        dev_loader = DataLoader(dev, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    def set_epoch(epoch):
        train.epoch = epoch
        generator.manual_seed(seed+epoch)

    set_epoch(0)
    return DataBundle(train_loader, dev_loader, metadata, set_epoch)


def build_loaders(config, *, seed):
    return _build(config, seed, refit=False)


def build_refit_loaders(config, *, seed):
    return _build(config, seed, refit=True)
