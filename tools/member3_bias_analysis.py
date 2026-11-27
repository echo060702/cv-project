"""Train-only metadata diagnostic, evaluated only on dev. Never fits image names or paths."""
import csv
import json
from pathlib import Path

import numpy as np

from team_eval import classification_metrics, sha


def main():
    split = list(csv.DictReader(Path('data/splits.csv').open(encoding='utf-8')))
    audit = {r['image_id']: r for r in csv.DictReader(Path('data/audit/images.csv').open(encoding='utf-8'))}
    train = [audit[r['image_id']] for r in split if r['split'] == 'train']
    dev = [audit[r['image_id']] for r in split if r['split'] == 'dev']
    formats = sorted({r['format'] for r in train})
    modes = sorted({r['mode'] for r in train})
    y = np.array([int(r['label']) for r in train])
    yd = np.array([int(r['label']) for r in dev])
    # Category probabilities are fitted exclusively on train, with Laplace smoothing.
    probabilities = {f: (sum(int(r['label']) for r in train if r['format'] == f)+1)/
                        (sum(r['format'] == f for r in train)+2) for f in formats}
    format_result = classification_metrics(yd, [probabilities.get(r['format'], (y.sum()+1)/(len(y)+2)) for r in dev])

    def features(rows, include_format):
        result = []
        for r in rows:
            w, h = int(r['width']), int(r['height'])
            v = [np.log(w), np.log(h), np.log(w/h), float(min(w,h)<224)]
            v += [float(r['mode'] == m) for m in modes]
            if include_format:
                v += [float(r['format'] == f) for f in formats]
            result.append(v)
        return np.array(result)

    results = {}
    for name, include_format in [('size_and_color', False), ('size_color_and_format', True)]:
        x, xd = features(train, include_format), features(dev, include_format)
        mean, scale = x.mean(0), x.std(0)
        scale[scale == 0] = 1
        x = np.c_[np.ones(len(x)), (x-mean)/scale]
        xd = np.c_[np.ones(len(xd)), (xd-mean)/scale]
        weights = np.zeros(x.shape[1])
        # Fixed optimizer and regularization; no dev tuning.
        for _ in range(2000):
            p = 1/(1+np.exp(-np.clip(x@weights,-40,40)))
            weights -= .05*(x.T@(p-y)/len(y)+.001*np.r_[0,weights[1:]])
        pd = 1/(1+np.exp(-np.clip(xd@weights,-40,40)))
        results[name] = dict(metrics=classification_metrics(yd,pd), weights=weights.tolist(),
                             train_mean=mean.tolist(), train_scale=scale.tolist())
    report = dict(scope='diagnostic only; train fitted, dev evaluated; no holdout or test',
                  train_count=len(train), dev_count=len(dev), format_probabilities=probabilities,
                  format_only=format_result, logistic_models=results, modes=modes, formats=formats,
                  features=['log_width','log_height','log_aspect_ratio','short_side_below_224','train_mode_one_hot','optional_train_format_one_hot'],
                  fitting=dict(iterations=2000,learning_rate=.05,l2=.001,dev_tuning=False),
                  manifest_sha256=sha('data/splits.csv'),audit_sha256=sha('data/audit/images.csv'),
                  source_sha256=sha(__file__), path_or_name_used_as_feature=False)
    out=Path('results/evaluation');out.mkdir(parents=True,exist_ok=True)
    (out/'metadata_bias.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'format_only':format_result,'other_models':{k:v['metrics'] for k,v in results.items()}},indent=2))


if __name__ == '__main__':
    main()
