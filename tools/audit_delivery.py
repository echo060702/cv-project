"""第一轮：职责/文件/配置/工程产物完整性核对。

两个阶段：
  --stage handoff  中途交接包（允许 G43 只跑了 N/20，校验进度快照与实际 run 一致）
  --stage final    六次跑满后的最终交付（要求 results/baselines 汇总与 6 套 20 epoch run）
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from cvdetect.contracts import validate_config
from cvdetect.engine import source_fingerprint

EXPERIMENTS = ('G', 'A')
SEEDS = (17, 29, 43)


def check_runs(root, progress):
    from tools.progress_report import read_progress
    recorded = {record['run']: record for record in progress['runs']}
    for run_name, record in recorded.items():
        current = read_progress(root/'runs/baselines'/run_name)
        for field in ('epochs_done', 'best_epoch', 'best_dev_accuracy', 'last_checkpoint_epoch', 'complete'):
            assert current[field] == record[field], (run_name, field, current[field], record[field])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('handoff', 'final'), default='final')
    args = parser.parse_args()
    stage = args.stage
    root = Path(__file__).resolve().parents[1]
    required = ['README.md', 'AGENTS.md', 'requirements.txt', '要求.md', '三人小组分工_待审核.md',
                '选题三_优化方案_待审核.md', '选题三_文献核查_20261122.md', '协作任务板.csv',
                'cvdetect/model.py', 'cvdetect/losses.py', 'cvdetect/contracts.py', 'cvdetect/engine.py',
                'cvdetect/train.py', 'cvdetect/inference.py', 'cvdetect/refit.py', 'cvdetect/__init__.py',
                'docs/模型训练_实施与验收计划.md', 'docs/架构与方法.md', 'docs/模型结构.svg',
                'docs/成员1_实施与验收计划.md', 'docs/成员1交付状态.md', 'docs/成员1接手指南.md',
                'docs/数据交接接口.md', 'docs/成员2接手指南.md', 'docs/成员3接手指南.md',
                'docs/模型训练_方法章节草稿.md', 'docs/预期结果与验收.md', 'docs/最终重训与结果追溯.md',
                'docs/官方来源与环境依据.md', 'docs/协作变更与贡献记录.md', 'docs/模型训练_工程交付状态.md',
                'docs/结果记录模板.csv', 'validation/smoke/report.json',
                '浏览器核验_20261121/ai_overview.txt', '浏览器核验_20261121/ai_data.txt',
                '浏览器核验_20261121/ai_rules.txt', 'tools/package_team.py', 'tools/check_archive.py',
                'tools/progress_report.py', 'tools/summarize_baselines.py',
                '00_交接说明_两位成员先读.md', 'docs/交接文档_当前进度与接手说明.md', 'docs/环境与已知问题.md',
                'team_data.py', 'tools/prepare_data.py', 'tools/verify_data_handoff.py',
                'data/raw/train_labels.csv', 'data/raw/sample_submission.csv',
                'data/audit/images.csv', 'data/audit/test_files.csv', 'data/statistics.json', 'data/splits.csv',
                'data/splits.sha256', 'data/splits.json', 'validation/budget_freeze.json',
                'validation/data_pipeline_report.json', 'validation/submission_template_source.json',
                'validation/baseline_execution_authorization.json', 'validation/G_seed29_duplicate_incident.json',
                'docs/数据审计报告.md', 'docs/数据与基线章节草稿.md', 'docs/成员1_GA正式实验执行与验收.md',
                'results/handoff/progress.json', 'results/handoff/progress.csv', 'tasks.json']
    required += [f'configs/{exp}.json' for exp in ('G', 'A', 'B', 'C', 'D', 'E', 'F')]
    required += [f'validation/smoke/synthetic_{exp}/{name}' for exp in ('B', 'C', 'D')
                 for name in ('config.json', 'metadata.json', 'environment.json', 'metrics.jsonl', 'best.pt', 'last.pt', 'dev_predictions.csv')]
    artifact_names = ('config.json', 'metadata.json', 'environment.json', 'metrics.jsonl', 'best.pt', 'last.pt', 'dev_predictions.csv')
    if stage == 'final':
        required += ['results/baselines/baseline_summary.json', 'results/baselines/per_run.csv',
                     'results/baselines/summary.csv', 'results/baselines/training_curves.png',
                     'results/baselines/training_curves.pdf', 'docs/G_A正式基线结果与分析.md',
                     'validation/baseline_delivery_tests_round1.txt']
        required += [f'runs/baselines/{exp}_seed{seed}/{name}' for exp in EXPERIMENTS for seed in SEEDS
                     for name in artifact_names]
    else:
        required += [f'runs/baselines/{exp}_seed{seed}/{name}' for exp in EXPERIMENTS for seed in (17, 29)
                     for name in artifact_names]
        required += ['runs/baselines/G_seed43/last.pt', 'runs/baselines/G_seed43/metrics.jsonl',
                     'validation/handoff_tests_round1.txt']
    missing = [name for name in required if not (root/name).is_file() or (root/name).stat().st_size == 0]
    if missing:
        raise AssertionError(f'missing required deliverables: {missing}')

    with (root/'协作任务板.csv').open(encoding='utf-8', newline='') as stream:
        tasks = {row['task_id']: row for row in csv.DictReader(stream)}
    assert tasks['D04']['owner'] == '成员1' and tasks['M07']['owner'] == '成员1'
    if stage == 'final':
        assert '六次各20epoch' in tasks['D04']['status']
        assert tasks['M08']['owner'] == '成员2' and 'B/C/D' in tasks['M08']['task']
    else:
        assert '已完成4次' in tasks['D04']['status']
        assert tasks['M08']['owner'] == '成员2'
    for task_id, task in tasks.items():
        if task_id.startswith('M') and task_id not in ('M07', 'M08'):
            assert task['owner'] == '成员2', task_id
        elif task_id.startswith('E'):
            assert task['owner'] == '成员3', task_id

    mirror = json.loads((root/'tasks.json').read_text(encoding='utf-8'))
    by_id = {item['task_id']: item for item in mirror['tasks']}
    assert set(by_id) == set(tasks), 'tasks.json out of sync with 协作任务板.csv'
    for task_id, task in tasks.items():
        assert by_id[task_id]['owner'] == task['owner'] and by_id[task_id]['task'] == task['task'], task_id
    for task_id in ('M04', 'M08'):
        assert by_id[task_id]['owner'] == '成员2' and 'B/C/D' in by_id[task_id]['task'], task_id
    assert by_id['D04']['owner'] == '成员1' and by_id['D04']['status'].startswith('已完成')

    roles = (root/'AGENTS.md').read_text(encoding='utf-8')
    if stage == 'final':
        assert '六次正式基线' in roles and 'G/A' in roles
    else:
        assert 'G17/A17/G29/A29' in roles and 'G43' in roles
        assert 'G43' in (root/'README.md').read_text(encoding='utf-8')

    python_paths = [p for folder in ('cvdetect', 'tests', 'tools') for p in (root/folder).rglob('*.py')] + [root/'team_data.py']
    for path in python_paths:
        ast.parse(path.read_text(encoding='utf-8'))
    for path in list(root.glob('*.md')) + list((root/'docs').glob('*.md')):
        if '\ufffd' in path.read_text(encoding='utf-8'):
            raise AssertionError(f'UTF8 replacement character: {path.name}')
    ET.parse(root/'docs/模型结构.svg')

    budget = json.loads((root/'validation/budget_freeze.json').read_text(encoding='utf-8'))
    recipes = []
    for exp in ('G', 'A', 'B', 'C', 'D', 'E', 'F'):
        config = json.loads((root/f'configs/{exp}.json').read_text(encoding='utf-8'))
        validate_config(config)
        assert config['status'] == 'frozen_budget'
        assert hashlib.sha256((root/f'configs/{exp}.json').read_bytes()).hexdigest() == budget['config_sha256'][exp]
        assert config['training']['epochs'] == 20
        recipe = dict(config['training'])
        recipe.pop('consistency')
        recipes.append(recipe)
    assert all(r == recipes[0] for r in recipes)

    smoke = json.loads((root/'validation/smoke/report.json').read_text(encoding='utf-8'))
    assert smoke['passed'] and smoke['environment']['source_sha256'] == source_fingerprint()
    assert all(item['data_kind'] == 'synthetic_engineering' for item in smoke['synthetic_fits'])

    data_report = json.loads((root/'validation/data_pipeline_report.json').read_text(encoding='utf-8'))
    assert data_report['passed'] and data_report['statistics_recomputed_train_only']
    template_source = json.loads((root/'validation/submission_template_source.json').read_text(encoding='utf-8'))
    assert template_source['official_bytes_verified']

    progress = json.loads((root/'results/handoff/progress.json').read_text(encoding='utf-8'))
    if stage == 'final':
        assert progress['completed_runs'] == 6 and progress['all_runs_complete'] and not progress['pending']
        assert all(item['epochs_done'] == 20 for item in progress['runs'])
    else:
        assert progress['completed_runs'] == 4 and progress['total_planned_runs'] == 6 and not progress['all_runs_complete']
        pending = {(item['experiment'], item['seed']): item for item in progress['pending']}
        assert pending[('G', 43)]['epochs_done'] == 6 and pending[('G', 43)]['status'] == 'partial_resumable'
        assert pending[('A', 43)]['status'] == 'not_started'
    check_runs(root, progress)

    result = {'round': 1, 'stage': stage, 'passed': True, 'required_files_checked': len(required),
              'python_files_parsed': len(python_paths), 'source_sha256': source_fingerprint(),
              'seven_shared_recipes': True, 'synthetic_artifacts_explicitly_labeled': True,
              'svg_xml_valid': True, 'data_pipeline_report': data_report,
              'official_submission_template_verified': True,
              'progress': {'completed_runs': progress['completed_runs'],
                           'runs': [{'run': record['run'], 'epochs_done': record['epochs_done'],
                                     'best_epoch': record['best_epoch'],
                                     'best_dev_accuracy': record['best_dev_accuracy']} for record in progress['runs']],
                           'pending': progress['pending']},
              'formal_GA_completed_runs': progress['completed_runs'],
              'formal_GA_summary_exists': (root/'results/baselines/baseline_summary.json').is_file(),
              'holdout_evaluated': False, 'test_pixels_decoded': False}
    if stage == 'final':
        test_log = (root/'validation/baseline_delivery_tests_round1.txt').read_text(encoding='utf-8')
        assert 'Ran 21 tests' in test_log and 'OK' in test_log
        assert progress['all_runs_complete']
        baselines = json.loads((root/'results/baselines/baseline_summary.json').read_text(encoding='utf-8'))
        assert baselines['passed'] and baselines['completed_runs'] == 6 and baselines['epochs_per_run'] == 20
        assert baselines['checkpoint_predictions_replayed']
        from tools.summarize_baselines import inspect_run
        for recorded in baselines['runs']:
            current, _ = inspect_run(root/recorded['run_directory'])
            assert current['artifact_sha256'] == recorded['artifact_sha256']
            assert recorded['checkpoint_replay']['passed'] and recorded['dev_count'] == 1200
        target = root/'validation/baseline_delivery_round1.json'
    else:
        test_log = (root/'validation/handoff_tests_round1.txt').read_text(encoding='utf-8')
        assert 'Ran 21 tests' in test_log and 'OK' in test_log
        target = root/'validation/handoff_round1.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
