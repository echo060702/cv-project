"""Audit B/C/D checkpoints and report development ablations against delivered A."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
import torch

from cvdetect.engine import evaluate_dev, seed_everything
from cvdetect.inference import load_detector
from team_data import build_loaders
from tools.run_member2 import EXPERIMENTS, ROOT, SEEDS, freeze_execution
from tools.summarize_baselines import inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-predictions', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    freeze = freeze_execution()
    with Path('data/splits.csv').open(encoding='utf-8', newline='') as stream:
        dev_labels = {r['image_id']: int(r['label']) for r in csv.DictReader(stream) if r['split'] == 'dev'}
    baseline = json.loads(Path('results/baselines/baseline_summary.json').read_text(encoding='utf-8'))
    records, histories = [], {}
    for seed in SEEDS:
        for experiment in EXPERIMENTS:
            directory = Path('runs/member2')/f'{experiment}_seed{seed}'
            record, history = inspect_run(directory)
            config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
            expected = json.loads(Path(f'configs/{experiment}.json').read_text(encoding='utf-8'))
            expected.update(seed=seed, adapter='team_data:build_loaders', adapter_source_sha256=freeze['data_adapter_sha256'])
            if config != expected or record['source_sha256'] != freeze['source_sha256'] or record['split_sha256'] != freeze['split_sha256']:
                raise ValueError(f'{directory}: run differs from execution freeze')
            with (directory/'dev_predictions.csv').open(encoding='utf-8', newline='') as stream:
                saved = {r['image_id']: r for r in csv.DictReader(stream)}
            if set(saved) != set(dev_labels) or any(int(saved[i]['label']) != y for i, y in dev_labels.items()):
                raise ValueError(f'{directory}: labels differ from frozen dev manifest')
            reference = json.loads(Path(f'runs/baselines/A_seed{seed}/metadata.json').read_text(encoding='utf-8'))
            metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            if metadata != reference:
                raise ValueError(f'{directory}: data metadata differs from delivered A')
            if args.verify_predictions:
                seed_everything(seed)
                bundle = build_loaders(config, seed=seed)
                model, checkpoint = load_detector(directory/'best.pt', device=args.device)
                if checkpoint['metadata'] != bundle.metadata:
                    raise ValueError('current data metadata differs from checkpoint')
                metrics, predictions = evaluate_dev(model, bundle.dev_loader, torch.device(args.device))
                max_error = max(abs(float(saved[r['image_id']]['prob_fake'])-r['prob_fake']) for r in predictions)
                if metrics['accuracy'] != record['dev_accuracy'] or not math.isclose(metrics['loss'], record['dev_bce'], abs_tol=1e-6, rel_tol=1e-6) or max_error > 1e-6:
                    raise ValueError(f'{directory}: checkpoint replay mismatch')
                record['checkpoint_replay'] = dict(passed=True, images=len(predictions), max_probability_error=max_error,
                                                   bce=metrics['loss'], device=args.device)
                del model, bundle, checkpoint
            records.append(record)
            histories[experiment, seed] = history
            print(f'verified {experiment} seed={seed}: dev={record["dev_accuracy"]:.6f}, epoch={record["best_epoch"]}', flush=True)

    summaries = []
    values = {(r['experiment'], r['seed']): r['dev_accuracy'] for r in baseline['runs'] if r['experiment'] == 'A'}
    values.update({(r['experiment'], r['seed']): r['dev_accuracy'] for r in records})
    for experiment in ('A', *EXPERIMENTS):
        subset = [r for r in (baseline['runs'] if experiment == 'A' else records) if r['experiment'] == experiment]
        summaries.append(dict(experiment=experiment, runs=3,
                              dev_accuracy_mean=statistics.mean(r['dev_accuracy'] for r in subset),
                              dev_accuracy_sample_std=statistics.stdev(r['dev_accuracy'] for r in subset),
                              dev_bce_mean=statistics.mean(r['dev_bce'] for r in subset),
                              parameters=subset[0]['parameters'],
                              origin='delivered member1 results; not retrained' if experiment == 'A' else 'member2 official runs'))
    contrasts = {}
    for name, weights in (
        ('B_minus_A_residual_without_JS', {'B': 1, 'A': -1}),
        ('C_minus_A_JS_without_residual', {'C': 1, 'A': -1}),
        ('D_minus_B_JS_with_residual', {'D': 1, 'B': -1}),
        ('D_minus_C_residual_with_JS', {'D': 1, 'C': -1}),
        ('D_minus_A_combined', {'D': 1, 'A': -1}),
        ('interaction_D_minus_B_minus_C_plus_A', {'D': 1, 'B': -1, 'C': -1, 'A': 1}),
    ):
        differences = [100*sum(weight*values[experiment, seed] for experiment, weight in weights.items()) for seed in SEEDS]
        contrasts[name] = dict(seeds=list(SEEDS), paired_seed_differences_pp=differences,
                               mean_pp=statistics.mean(differences), sample_std_pp=statistics.stdev(differences))
    output = Path('results/member2')
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ('per_run.csv', [{k: v for k, v in r.items() if k not in ('artifact_sha256', 'checkpoint_replay')} for r in records]),
        ('summary.csv', summaries),
        ('paired_contrasts.csv', [dict(contrast=name, **record) for name, record in contrasts.items()]),
    ):
        with (output/name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(passed=True, completed_runs=9, epochs_per_run=20, threshold=0.5,
                  checkpoint_selection=freeze['selection'], checkpoint_predictions_replayed=args.verify_predictions,
                  scope='clean single center patch on dev used for checkpoint selection; not independent test performance',
                  source_sha256=freeze['source_sha256'], split_sha256=freeze['split_sha256'],
                  baseline_report_sha256=hashlib.sha256(Path('results/baselines/baseline_summary.json').read_bytes()).hexdigest(),
                  timing_caveat='epoch seconds are wall-clock measurements and can include machine suspension or interruptions; not GPU compute time',
                  holdout_evaluated=False, test_pixels_decoded=False, final_refit_performed=False,
                  runs=records, summary=summaries, contrasts=contrasts,
                  pending='member3 E/F capacity controls, independent evaluation and freeze; then member2 final refit')
    (output/'ablation_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    colors = dict(A='#666666', B='#0072B2', C='#D55E00', D='#009E73')
    for experiment in ('A', *EXPERIMENTS):
        group = []
        for seed in SEEDS:
            history = histories.get((experiment, seed))
            if history is None:
                history = [json.loads(line) for line in Path(f'runs/baselines/A_seed{seed}/metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            group.append(history)
        for axis, metric in zip(axes[:2], ('dev_loss', 'dev_accuracy')):
            array = np.array([[r[metric] for r in history] for history in group])
            mean, std = array.mean(axis=0), array.std(axis=0, ddof=1)
            axis.plot(range(1, 21), mean, label=experiment, color=colors[experiment])
            axis.fill_between(range(1, 21), mean-std, mean+std, alpha=0.12, color=colors[experiment])
    for axis, label in zip(axes[:2], ('Dev BCE', 'Dev accuracy')):
        axis.set(xlabel='Epoch', ylabel=label, xlim=(1, 20))
        axis.grid(alpha=0.2)
        axis.legend()
    axes[2].bar(range(4), [s['dev_accuracy_mean'] for s in summaries],
                yerr=[s['dev_accuracy_sample_std'] for s in summaries], color=list(colors.values()), alpha=0.7, capsize=4)
    for index, experiment in enumerate(colors):
        axes[2].scatter([index-0.08, index, index+0.08], [values[experiment, seed] for seed in SEEDS], color='black', s=18, zorder=3)
    axes[2].set(xticks=range(4), xticklabels=list(colors), ylabel='Selected dev accuracy', ylim=(0, 1), title='Mean +/- sample SD; 3 seeds')
    axes[2].grid(axis='y', alpha=0.2)
    fig.savefig(output/'ablation_curves.png', dpi=200)
    fig.savefig(output/'ablation_curves.pdf')
    plt.close(fig)

    lines = ['# 成员2 B/C/D 正式开发实验与消融', '',
             '九次均为官方 train/dev、随机初始化、20 epoch。每组按开发集选出最佳 checkpoint；下列为同一开发集上的选择后成绩，不是独立测试成绩。', '',
             '| 组 | 结构 | JS | 参数量 | dev Accuracy（均值 ± 样本标准差） |', '|---|---|---|---:|---:|']
    structures = {'A': ('单 RGB', '0'), 'B': ('RGB + 残差', '0'), 'C': ('单 RGB', '0.1'), 'D': ('RGB + 残差', '0.1')}
    for summary in summaries:
        exp = summary['experiment']
        lines.append(f'| {exp} | {structures[exp][0]} | {structures[exp][1]} | {summary["parameters"]:,} | {100*summary["dev_accuracy_mean"]:.2f}% ± {100*summary["dev_accuracy_sample_std"]:.2f}% |')
    lines += ['', 'A 为成员1已交付的三种子结果，本轮未重新训练或改写。', '',
              '| 配对差值（同种子） | 均值（百分点） | 样本标准差（百分点） |', '|---|---:|---:|']
    for name, contrast in contrasts.items():
        lines.append(f'| {name} | {contrast["mean_pp"]:+.3f} | {contrast["sample_std_pp"]:.3f} |')
    lines += ['', '差值用于描述三次重复的方向和波动；不将三种子差值写成显著性结论。B 相对 A、D 相对 C 同时增加了辅助网络，必须等成员3完成同容量 E/F，才可进一步区分残差输入与参数量的作用。', '',
              '所有组共享正常/JPEG两视图监督。无 JS 组仅令系数为零；C−A、D−B检验一致性项的影响。交互项为 D−B−C+A，不应仅凭 D−A 宣称两项存在协同。', '',
              '真实图全部 JPEG、AI 图全部 PNG，数据源/格式可能构成捷径。高准确率不证明跨来源泛化，也不证明重编码稳健；成员3需用冻结后的独立评价补充。', '',
              '完整证据：`results/member2/ablation_summary.json`、`per_run.csv`、`paired_contrasts.csv`、`ablation_curves.png`；逐 run 的日志、环境、best/last 和开发逐图概率在 `runs/member2/`。', '',
              '最终全量模型尚未训练：等待 E/F、模型选择摘要和评估冻结；不提前使用 holdout，也不生成官方提交 CSV。', '']
    Path('docs/成员2_BCD结果与分析.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
