"""只在训练入口核对一次配置、划分元数据与首批输入。"""
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class DataBundle:
    train_loader: object
    dev_loader: object
    metadata: dict
    set_epoch: Callable[[int], None]


def validate_metadata(meta, *, refit=False):
    required = {'data_kind', 'data_revision', 'split_sha256', 'mean', 'std', 'splits'}
    if not required.issubset(meta):
        raise ValueError(f'metadata missing: {sorted(required - set(meta))}')
    if meta['data_kind'] not in ('official', 'synthetic_engineering'):
        raise ValueError('data_kind must be official or synthetic_engineering')
    if len(meta['split_sha256']) != 64 or any(c not in '0123456789abcdef' for c in meta['split_sha256']):
        raise ValueError('split_sha256 must be lowercase SHA256')
    if len(meta['mean']) != 3 or len(meta['std']) != 3 or any(v <= 0 for v in meta['std']):
        raise ValueError('normalization requires 3 means and positive stds')
    splits = meta['splits']
    if set(splits) != {'train', 'dev', 'holdout'}:
        raise ValueError('split keys must be exactly train/dev/holdout; official test is forbidden')
    seen_ids, seen_groups = set(), set()
    for name, items in splits.items():
        ids, groups = items['image_ids'], items['group_ids']
        if len(ids) != len(groups) or len(ids) != len(set(ids)):
            raise ValueError(f'{name}: image/group count or duplicate image IDs')
        if (name == 'train' or (name == 'dev' and not refit)) and not ids:
            raise ValueError(f'{name}: empty split')
        if refit and name != 'train' and ids:
            raise ValueError('refit metadata must put all labeled images in train; dev/holdout empty')
        if seen_ids.intersection(ids):
            raise ValueError(f'{name}: image leakage')
        if seen_groups.intersection(groups):
            raise ValueError(f'{name}: group leakage')
        seen_ids.update(ids)
        seen_groups.update(groups)


def validate_config(config):
    if config['experiment'] not in ('G', 'A', 'B', 'C', 'D', 'E', 'F'):
        raise ValueError('unknown experiment')
    expected = {'G': ('global', 'none', 0.), 'A': ('local', 'none', 0.),
                'B': ('local', 'residual', 0.), 'C': ('local', 'none', 0.1),
                'D': ('local', 'residual', 0.1), 'E': ('local', 'rgb', 0.), 'F': ('local', 'rgb', 0.1)}
    actual = (config['data']['mode'], config['model']['branch'], config['training']['consistency'])
    if actual != expected[config['experiment']]:
        raise ValueError(f'experiment switches inconsistent: expected {expected[config["experiment"]]}')
    train = config['training']
    if min(train['epochs'], train['batch_size'], config['model']['width']) < 2:
        raise ValueError('epochs, batch_size, model width must be >= 2')
    if config['data']['image_size'] < 16 or train['learning_rate'] <= 0 or train['weight_decay'] < 0:
        raise ValueError('invalid size/optimizer configuration')
    if not 0 < train['warmup_fraction'] <= 1 or train['workers'] < 0:
        raise ValueError('invalid warmup/workers')
    if not isinstance(config['seed'], int) or not 0 <= config['seed'] < 2**31:
        raise ValueError('invalid seed')


def validate_bundle(bundle, config):
    validate_metadata(bundle.metadata)
    size = config['data']['image_size']
    for name, loader in (('train', bundle.train_loader), ('dev', bundle.dev_loader)):
        if len(loader) == 0:
            raise ValueError(f'{name}: empty loader')
        batch = next(iter(loader))
        required = {'image', 'image_jpeg', 'label', 'image_id', 'group_id'}
        if not required.issubset(batch):
            raise ValueError(f'{name}: missing batch keys {required - set(batch)}')
        x, y = batch['image'], batch['label']
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, size, size) or batch['image_jpeg'].shape != x.shape:
            raise ValueError(f'{name}: expected raw RGB NCHW {size} paired views')
        if not x.is_floating_point() or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError(f'{name}: image must be finite float in [0,1]')
        jpeg = batch['image_jpeg']
        if not jpeg.is_floating_point() or not torch.isfinite(jpeg).all() or jpeg.min() < 0 or jpeg.max() > 1:
            raise ValueError(f'{name}: image_jpeg must be finite float in [0,1]')
        if y.shape != (len(x),) or not ((y == 0) | (y == 1)).all():
            raise ValueError(f'{name}: labels must be binary shape [B]')
        members = bundle.metadata['splits'][name]
        membership = dict(zip(members['image_ids'], members['group_ids']))
        if any(membership.get(i) != g for i, g in zip(batch['image_id'], batch['group_id'])):
            raise ValueError(f'{name}: first batch disagrees with frozen split metadata')
