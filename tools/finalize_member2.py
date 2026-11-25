"""After nine audited runs, refresh handoff state, run checks, and package evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.run_member2 import ROOT, freeze_execution


def main():
    os.chdir(ROOT)
    freeze_execution()
    summary = json.loads(Path('results/member2/ablation_summary.json').read_text(encoding='utf-8'))
    if not summary['passed'] or summary['completed_runs'] != 9 or not summary['checkpoint_predictions_replayed']:
        raise ValueError('all nine complete official runs and checkpoint replays are required')
    if not all(r['checkpoint_replay']['passed'] for r in summary['runs']):
        raise ValueError('one or more checkpoint replays failed')
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUTF8='1')
    tests_path = Path('validation/member2_tests_final.txt')
    with tests_path.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-X', 'utf8', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    data = json.loads(Path('validation/member2_data_pipeline.json').read_text(encoding='utf-8'))
    method = json.loads(Path('validation/member2_method_checks.json').read_text(encoding='utf-8'))
    if not data['passed'] or not method['passed']:
        raise ValueError('data/method verification must pass')
    tasks_path = Path('tasks.json')
    tasks = json.loads(tasks_path.read_text(encoding='utf-8'))
    for task in tasks['tasks']:
        if task['task_id'] in ('M04', 'M08'):
            task['status'] = '已完成_BCD九次各20epoch_九套checkpoint开发预测重放通过'
        if task['task_id'] == 'M06':
            task['status'] = '已完成_方法核验_中英文方法稿_BCD开发消融与局限分析'
    tasks_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    group = {row['experiment']: row for row in summary['summary']}
    contrasts = summary['contrasts']
    interpretation = ['\n## 本轮结果的解释', '']
    for label, key in (('残差分支B相对局部基线A', 'B_minus_A_residual_without_JS'),
                       ('单RGB中加入JS：C−A', 'C_minus_A_JS_without_residual'),
                       ('残差模型中加入JS：D−B', 'D_minus_B_JS_with_residual'),
                       ('两因素交互项：D−B−C+A', 'interaction_D_minus_B_minus_C_plus_A')):
        row = contrasts[key]
        differences = ' / '.join(f'{value:+.2f}' for value in row['paired_seed_differences_pp'])
        interpretation.append(f'- {label}：平均变化{row["mean_pp"]:+.2f}个百分点，三个种子分别为{differences}个百分点。')
    interpretation += ['',
                       '本轮均值最高的是B。C−A与D−B均为负，说明在这套冻结预算和正常单patch开发评价下，JS没有改善平均开发准确率；不因方法预期而忽略负结果。是否改善重编码稳健性，要由成员3执行既定退化评价后判断。', '',
                       'B/D相对A/C增加了93,600个参数，约3.35%；当前不能把其全部提升归因于残差输入。B−E、D−F是后续判断残差相对第二路RGB是否有益所需的同容量对照。三种子差值描述方向和重复波动，不等同统计显著性或跨来源泛化。', '',
                       '时长取自逐epoch墙钟记录。B_seed17第12轮包含较长中断；保留原始时间并单列每run的中位数、最小/最大值，不把含中断的总秒数解释为纯GPU训练成本。', '']
    with Path('docs/成员2_BCD结果与分析.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(interpretation))
    status = ['# 成员2交付状态', '',
              'B/C/D 各三个种子17/29/43、每次20轮，合计9次180轮全部完成。九套最佳权重已逐一重载，每套重放1200张dev，核对Accuracy、BCE与逐图概率，共10800次图像预测。', '',
              '| 组 | dev Accuracy 均值 | 样本标准差 |', '|---|---:|---:|']
    for experiment in ('A', 'B', 'C', 'D'):
        row = group[experiment]
        status.append(f'| {experiment} | {100*row["dev_accuracy_mean"]:.2f}% | {100*row["dev_accuracy_sample_std"]:.2f}% |')
    status += ['', 'A为成员1已有结果。本轮保留共享模型、损失、数据处理、冻结划分和七组配置的原始字节，新增成员2执行、审计、交付工具与方法文稿。', '',
               '这些是用于选择checkpoint的开发集成绩。B相对A的变化包含辅助分支容量；待成员3的E/F后才能进一步判断残差输入本身的作用。JS是否有益按C−A、D−B如实报告，不预设正面结论。', '',
               '交付内容：`runs/member2/`九套best/last、逐轮日志、config/metadata/environment和dev逐图预测；`results/member2/`汇总/逐run/配对差值/曲线；`docs/模型训练_方法章节草稿.md`、`docs/member2_method.tex`和方法讲稿；`validation/`真实核验报告。', '',
               '交付包：`dist/member2_BCD_delivery.zip`，含代码、文稿、九套训练权重和证据，不含原图或Python环境。原图已在本机恢复，接收者按`docs/仓库说明_数据与权重获取.md`从官方固定版本获取；A/G权重仍需原Release，本轮不重新训练。', '',
               '仍待后续：成员3完成E/F、独立评价和评估冻结，提供真实selection_summary.json与evaluation_freeze.json后，成员2再进行全量有标签refit。当前没有最终全量final.pt，也未执行holdout确认或官方测试推理。', '',
               '无需重跑九次实验。成员3下一步接收本包与官方数据，按既定规则完成其评估和E/F；最终模型选择由完整实验和冻结文件决定，不凭本轮最高dev直接确定。', '']
    Path('docs/成员2交付状态.md').write_text('\n'.join(status), encoding='utf-8')
    Path('README.md').write_text('''# AI生成图像识别：成员2 B/C/D 交付

成员2已完成B/C/D三组各三个种子（17/29/43），每次20轮，共9次180轮。九套最佳权重已分别重放1200张开发图，逐图概率、Accuracy和BCE与保存记录一致。

- 当前交付状态：[成员2交付状态](docs/成员2交付状态.md)
- 实验结果与局限：[B/C/D结果与分析](docs/成员2_BCD结果与分析.md)
- 环境、运行和恢复：[执行与交付](docs/成员2执行与交付.md)
- 方法：[中文稿](docs/模型训练_方法章节草稿.md)、[英文LaTeX片段](docs/member2_method.tex)、[约3分钟讲稿](docs/成员2_3分钟方法讲稿.md)
- 原成员1交接资料：[原始README](docs/成员1_README_原始交接.md)、[G/A结果](docs/G_A正式基线结果与分析.md)

## 结果与产物

`runs/member2/`包含九套config、metadata、environment、20轮metrics.jsonl、best.pt、last.pt和dev_predictions.csv。`results/member2/`包含逐run表、三种子汇总、配对差值、曲线及ablation_summary.json。`validation/member2_delivery_round1.json`和`member2_tests_final.txt`记录本轮实测验收。

交付包`dist/member2_BCD_delivery.zip`包含代码、方法文稿、九套权重和证据，不含图像或Python环境。其SHA256和ZIP逐文件核验报告在旁边。接收者获取官方图像与原成员1 G/A权重的步骤见[数据与权重获取](docs/仓库说明_数据与权重获取.md)。本机已经恢复官方图像，不必重新下载。

## 核验与复算

使用项目内Python3.12虚拟环境：

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -v
.\\.venv\\Scripts\\python.exe -X utf8 -m tools.member2_status
.\\.venv\\Scripts\\python.exe -X utf8 -m tools.summarize_member2 --verify-predictions
.\\.venv\\Scripts\\python.exe -X utf8 -m tools.package_member2 --verify dist/member2_BCD_delivery.zip
```

模型、损失、训练器、数据处理和七组冻结配置保留共享工程的原始字节；本轮新增成员2实验执行、审计、交付工具与中英文文稿。全部正式训练从随机初始化开始，预算仍为20epoch/224/width32/batch16/workers0，未重跑G/A。

## 后续依赖

成员3仍需完成E/F同容量对照、独立评价及评估冻结，再交付真实selection_summary.json与evaluation_freeze.json。之后成员2才执行全量有标签refit。当前不包含final.pt或官方预测CSV，未使用holdout确认或官方test推理。

本轮dev参与checkpoint选择，不能作为独立测试成绩。真实类全JPEG、AI类全PNG的偏差需要结合成员3的诊断；残差的贡献要结合E/F，不能只依据B高于A或D高于C判断。
''', encoding='utf-8')
    with Path('docs/协作变更与贡献记录.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 成员2九次正式实验完成核验\n\n'
                     '- B/C/D各三个种子、各20epoch全部完成；九套best.pt重载后各重放1200张dev，逐图概率、Accuracy和BCE与记录一致。\n'
                     '- results/member2/ablation_summary.json、summary.csv、per_run.csv、paired_contrasts.csv和曲线为真实完成后生成；分析在docs/成员2_BCD结果与分析.md。A仍为成员1交付的结果，未重训或改写。\n'
                     '- 最终交付自测与第一轮核验见validation/member2_tests_final.txt及member2_delivery_round1.json；ZIP逐文件哈希和九套产物检查见dist/member2_BCD_delivery.verification.json。\n'
                     '- E/F、独立评价、评估冻结、最终全量模型及官方CSV仍待后续，不将本轮开发成绩描述为独立确认成绩。\n')
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.member2_status'], cwd=ROOT, env=environment, check=True)
    evidence = ['validation/member2_execution_freeze.json', 'validation/member2_data_pipeline.json',
                'validation/member2_method_checks.json', 'validation/member2_tests_final.txt',
                'results/member2/ablation_summary.json']
    report = dict(passed=True, complete_official_runs=9, epochs_per_run=20, checkpoint_replays=9,
                  dev_predictions_replayed=10800, full_test_suite_passed=True,
                  holdout_evaluated=False, test_pixels_decoded=False, final_refit_performed=False,
                  evidence_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in evidence})
    Path('validation/member2_delivery_round1.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.package_member2'], cwd=ROOT, env=environment, check=True)
    print('Nine official runs, checkpoint replay, final checks and delivery ZIP completed.', flush=True)


if __name__ == '__main__':
    main()
