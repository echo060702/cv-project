"""Run a resumable dev-only evaluation matrix; freeze only after every job is verified."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from cvdetect.engine import source_fingerprint
from team_eval import CONDITIONS, classification_metrics, evaluate, paired_group_bootstrap, sha


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path('results/evaluation/dev_v1')


def plan():
    jobs = []
    for exp in 'GABCDEF':
        category = 'baselines' if exp in 'GA' else 'member2' if exp in 'BCD' else 'member3'
        for seed in (17, 29, 43):
            checkpoint = f'runs/{category}/{exp}_seed{seed}/best.pt'
            for patches, condition in [(1,c) for c in CONDITIONS]+([(5,'clean')] if exp != 'G' else []):
                jobs.append(dict(key=f'{exp}_seed{seed}_p{patches}_{condition}',experiment=exp,seed=seed,
                                 patches=patches,condition=condition,checkpoint=checkpoint,
                                 checkpoint_sha256=sha(checkpoint)))
    return dict(version=1,scope='dev only; no holdout/test',jobs=jobs,
                manifest_sha256=sha('data/splits.csv'),statistics_sha256=sha('data/statistics.json'),
                evaluator_sha256=sha('team_eval.py'),source_sha256=source_fingerprint(),
                selection_rule='highest 3-seed mean clean dev accuracy; ties: fewer patches, fewer parameters, alphabetic experiment',
                threshold=.5,aggregation='mean_probability',final_seed=17,proposed_refit_epochs=20,
                perturbation_order='EXIF transpose, RGB, global resize or reflection pad; full-image JPEG subsampling=0 before fixed crop; scale05: crop then half-size bilinear then restore bilinear',
                five_patches='center, top-left, top-right, bottom-left, bottom-right; no flips',
                pressure_conditions=list(CONDITIONS[1:]))


def cached(job, protocol):
    folder = OUTPUT/job['key']
    report = json.loads((folder/'metrics.json').read_text())
    for key, expected in [('checkpoint_sha256',job['checkpoint_sha256']),('manifest_sha256',protocol['manifest_sha256']),
                          ('evaluator_sha256',protocol['evaluator_sha256']),('source_sha256',protocol['source_sha256']),
                          ('patches',job['patches']),('condition',job['condition']),('split','dev')]:
        if report[key] != expected:
            raise ValueError(f'{job["key"]}: cached {key} mismatch')
    if sha(folder/'predictions.csv') != report['predictions_sha256']:
        raise ValueError('cached predictions checksum mismatch')
    rows = list(csv.DictReader((folder/'predictions.csv').open(encoding='utf-8')))
    manifest = list(csv.DictReader(Path('data/splits.csv').open(encoding='utf-8')))
    expected = {r['image_id']:(r['group_id'],int(r['label'])) for r in manifest if r['split']=='dev'}
    if len(rows)!=len(expected) or {r['image_id']:(r['group_id'],int(r['label'])) for r in rows}!=expected:
        raise ValueError('cached dev inventory mismatch')
    recomputed = classification_metrics([int(r['label']) for r in rows],[float(r['prob_fake']) for r in rows])
    if recomputed != report['metrics']:
        raise ValueError('cached metrics mismatch')
    return report, rows


def finalize(protocol):
    # All jobs are verified before writing either selection or freeze output.
    verified = {j['key']:cached(j,protocol) for j in protocol['jobs']}
    summary_rows = []
    for exp in 'GABCDEF':
        for patches, condition in [(1,c) for c in CONDITIONS]+([(5,'clean')] if exp!='G' else []):
            reports = [verified[f'{exp}_seed{s}_p{patches}_{condition}'][0] for s in (17,29,43)]
            summary_rows.append(dict(experiment=exp,patches=patches,condition=condition,
                accuracy_mean=statistics.mean(r['metrics']['accuracy'] for r in reports),
                accuracy_sample_std=statistics.stdev(r['metrics']['accuracy'] for r in reports),
                roc_auc_mean=statistics.mean(r['metrics']['roc_auc'] for r in reports),
                wall_seconds_mean=statistics.mean(r['wall_seconds'] for r in reports)))
    errors = []
    for exp in 'GABCDEF':
        clean = {r['image_id']:r for r in verified[f'{exp}_seed17_p1_clean'][1]}
        for condition in CONDITIONS:
            for row in verified[f'{exp}_seed17_p1_{condition}'][1]:
                predicted = int(float(row['prob_fake']) >= .5)
                initial = int(float(clean[row['image_id']]['prob_fake']) >= .5)
                if predicted != int(row['label']) or predicted != initial:
                    errors.append(dict(experiment=exp,seed=17,condition=condition,**row,
                        clean_probability=clean[row['image_id']]['prob_fake'],
                        prediction=predicted,incorrect=predicted!=int(row['label']),flipped=predicted!=initial))
    candidates = []
    for exp in 'GABCDEF':
        for patches in ([1] if exp=='G' else [1,5]):
            reports=[verified[f'{exp}_seed{s}_p{patches}_clean'][0] for s in (17,29,43)]
            config=reports[0]['config']
            params=2795297 if config['model']['branch']=='none' else 2888897
            candidates.append(dict(experiment=exp,patches=patches,parameters=params,
                                   mean=statistics.mean(x['metrics']['accuracy'] for x in reports),
                                   sample_std=statistics.stdev(x['metrics']['accuracy'] for x in reports),
                                   mean_wall_seconds=statistics.mean(x['wall_seconds'] for x in reports)))
    selected=sorted(candidates,key=lambda x:(-x['mean'],x['patches'],x['parameters'],x['experiment']))[0]
    contrasts={}
    for a,b in [('B','E'),('D','F'),('D','B'),('F','E'),('B','A')]:
        for condition in CONDITIONS:
            contrasts[f'{a}-{b}/{condition}']=paired_group_bootstrap(
                verified[f'{a}_seed17_p1_{condition}'][1],verified[f'{b}_seed17_p1_{condition}'][1])
    selection=dict(selected=selected,candidates=candidates,selection_rule=protocol['selection_rule'],
                   protocol_sha256=sha(OUTPUT/'protocol.json'),seed17_paired_group_bootstrap=contrasts,
                   runner_sha256=sha(__file__),pressure_summary=summary_rows,
                   caveat='development selection bias and training-environment confounding remain',holdout_evaluated=False)
    checkpoint_jobs=[j for j in protocol['jobs'] if j['patches']==1 and j['condition']=='clean']
    freeze=dict(version=1,selected_experiment=selected['experiment'],patches=selected['patches'],
                frozen_at_utc=datetime.now(timezone.utc).isoformat(),runner_sha256=sha(__file__),
                threshold=.5,aggregation='mean_probability',seeds=[17,29,43],final_seed=17,
                proposed_refit_epochs=20,checkpoints=checkpoint_jobs,
                pressure_conditions=protocol['pressure_conditions'],protocol_sha256=sha(OUTPUT/'protocol.json'),
                source_sha256=protocol['source_sha256'],evaluator_sha256=protocol['evaluator_sha256'],
                split_sha256=protocol['manifest_sha256'],statistics_sha256=protocol['statistics_sha256'],
                holdout_policy='one confirmation after freeze; no model, threshold or inference retuning',
                holdout_evaluated=False,test_pixels_decoded=False)
    selection_path=Path('results/selection_summary.json')
    freeze_path=Path('results/evaluation_freeze.json')
    if selection_path.exists() or freeze_path.exists():
        raise ValueError('selection/freeze already exists; do not overwrite a frozen decision')
    for name, rows in [('dev_summary.csv',summary_rows),('seed17_errors_and_flips.csv',errors)]:
        if rows:
            with (OUTPUT/name).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selection_path.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    freeze['selection_summary_sha256']=sha(selection_path)
    freeze_path.write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    print('Frozen dev-selected method:',json.dumps(selected),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--finalize-only',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT);torch.set_num_threads(4)
    protocol=plan();OUTPUT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:
            raise ValueError('evaluation inputs changed; preserve prior results and use a new protocol version')
    else:path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.prepare_only:
        print(f'Prepared {len(protocol["jobs"])} dev jobs; holdout/test untouched.');return
    if args.finalize_only:
        finalize(protocol);return
    lock=OUTPUT/'.runner.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid()),f)
    try:
        for i,job in enumerate(protocol['jobs'],1):
            folder=OUTPUT/job['key']
            if folder.exists():cached(job,protocol)
            else:evaluate(job['checkpoint'],folder,device=args.device,patches=job['patches'],condition=job['condition'])
            print(f'{i}/{len(protocol["jobs"])} verified: {job["key"]}',flush=True)
        finalize(protocol)
    finally:lock.unlink()


if __name__=='__main__':
    main()
