import math

import torch
from torch.nn import functional as F


def js_divergence(logits_a, logits_b):
    # 在 log 空间计算 Bernoulli 分布，避免饱和 sigmoid 后出现 log(0)。
    a, b = logits_a.float(), logits_b.float()
    log_a = torch.stack((F.logsigmoid(-a), F.logsigmoid(a)), dim=-1)
    log_b = torch.stack((F.logsigmoid(-b), F.logsigmoid(b)), dim=-1)
    log_m = torch.logaddexp(log_a, log_b) - math.log(2)
    return (0.5 * (log_a.exp() * (log_a-log_m) + log_b.exp() * (log_b-log_m)).sum(dim=-1)).mean()


def paired_loss(logits_a, logits_b, labels, coefficient):
    supervised = 0.5 * (F.binary_cross_entropy_with_logits(logits_a.float(), labels.float()) +
                        F.binary_cross_entropy_with_logits(logits_b.float(), labels.float()))
    consistency = js_divergence(logits_a, logits_b)
    return supervised + coefficient * consistency, {'supervised': supervised.detach(), 'js': consistency.detach()}
