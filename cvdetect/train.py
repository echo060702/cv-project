import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

from .engine import run_training


def main():
    parser = argparse.ArgumentParser(description='成员1公共训练器；正式数据适配器由成员2交付')
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter', required=True, help='module:function, returning DataBundle')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--run-epochs', type=int, help='仅执行若干 epoch，保持总调度长度以便恢复')
    for option, kind in (('seed', int), ('epochs', int), ('batch-size', int), ('width', int), ('image-size', int), ('workers', int)):
        parser.add_argument('--'+option, type=kind)
    parser.add_argument('--data-root')
    parser.add_argument('--manifest')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    for key in ('epochs', 'batch_size', 'workers'):
        if getattr(args, key) is not None:
            config['training'][key] = getattr(args, key)
    if args.seed is not None:
        config['seed'] = args.seed
    if args.width is not None:
        config['model']['width'] = args.width
    for argument, key in (('image_size', 'image_size'), ('data_root', 'root'), ('manifest', 'manifest')):
        if getattr(args, argument) is not None:
            config['data'][key] = getattr(args, argument)
    module_name, function_name = args.adapter.split(':', 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    config['adapter'] = args.adapter
    config['adapter_source_sha256'] = hashlib.sha256(Path(inspect.getfile(factory)).read_bytes()).hexdigest()
    bundle = factory(config, seed=config['seed'])
    run_training(config, bundle, args.output, device=args.device, resume=args.resume, run_epochs=args.run_epochs)


if __name__ == '__main__':
    main()
