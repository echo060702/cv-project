"""仅工程测试：亮度可分的合成 RGB 与真实 Pillow JPEG 重编码，不是官方数据。"""
import hashlib
import io
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from cvdetect.contracts import DataBundle


class SyntheticPairs(Dataset):
    def __init__(self, count, size, split, seed):
        self.records = []
        generator = torch.Generator().manual_seed(seed)
        for index in range(count):
            label = index % 2
            x = (torch.rand(3, size, size, generator=generator)*0.12 + (0.12 if label == 0 else 0.74)).clamp(0, 1)
            rgb = (x.permute(1, 2, 0).numpy()*255).astype('uint8')
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format='JPEG', quality=75)
            jpeg = torch.from_numpy(np.array(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'), copy=True)).permute(2, 0, 1).float()/255
            self.records.append({'image': x, 'image_jpeg': jpeg, 'label': float(label),
                                 'image_id': f'synthetic-{split}-{index}', 'group_id': f'synthetic-{split}-{index}'})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def build_loaders(config, *, seed):
    size = config['data']['image_size']
    train = SyntheticPairs(32, size, 'train', seed)
    dev = SyntheticPairs(16, size, 'dev', seed+1)
    splits = {name: {'image_ids': [r['image_id'] for r in dataset.records], 'group_ids': [r['group_id'] for r in dataset.records]}
              for name, dataset in (('train', train), ('dev', dev))}
    splits['holdout'] = {'image_ids': [], 'group_ids': []}
    metadata = {'data_kind': 'synthetic_engineering', 'data_revision': f'brightness-fixture-v1-seed{seed}-size{size}',
                'split_sha256': hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
                'mean': [0.5]*3, 'std': [0.5]*3, 'splits': splits}
    generator = torch.Generator()
    train_loader = DataLoader(train, batch_size=config['training']['batch_size'], shuffle=True, generator=generator, num_workers=0)
    dev_loader = DataLoader(dev, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    return DataBundle(train_loader, dev_loader, metadata, lambda epoch: generator.manual_seed(seed+epoch))
