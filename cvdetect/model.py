"""无外部权重的轻量 ResNet-18 型骨干与严格同容量辅助分支。"""
import torch
from torch import nn
from torch.nn import functional as F


def highpass(x):
    return x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, stride=1)


class BasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outgoing),
        )
        self.shortcut = (nn.Identity() if incoming == outgoing and stride == 1 else
                         nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)))

    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x), inplace=True)


class Detector(nn.Module):
    def __init__(self, width=32, branch='none', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        if branch not in ('none', 'residual', 'rgb') or width < 2:
            raise ValueError('branch must be none/residual/rgb; width >= 2')
        self.branch = branch
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        layers = [nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        incoming = width
        for stage, outgoing in enumerate((width, width*2, width*4, width*8)):
            layers.extend([BasicBlock(incoming, outgoing, 1 if stage == 0 else 2), BasicBlock(outgoing, outgoing)])
            incoming = outgoing
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.backbone = nn.Sequential(*layers)
        if branch != 'none':
            aux = []
            incoming = 3
            for outgoing in (width, width*2, width*4):
                aux.extend([nn.Conv2d(incoming, outgoing, 3, 2, 1, bias=False), nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True)])
                incoming = outgoing
            aux.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
            self.auxiliary = nn.Sequential(*aux)
        self.classifier = nn.Linear(width * (8 if branch == 'none' else 12), 1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, image):
        normalized = (image - self.mean) / self.std
        features = self.backbone(normalized)
        if self.branch != 'none':
            # 先从原始 [0,1] 张量计算残差，再按同一固定通道尺度归一化。
            auxiliary = highpass(image) / self.std if self.branch == 'residual' else normalized
            features = torch.cat((features, self.auxiliary(auxiliary)), dim=1)
        return self.classifier(features).squeeze(1)


def build_model(config, metadata):
    return Detector(width=config['model']['width'], branch=config['model']['branch'],
                    mean=metadata['mean'], std=metadata['std'])
