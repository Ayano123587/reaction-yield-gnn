"""Input-only local molecular fingerprints and a frozen grouped comparison."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from .data import canonical
from .quantity_graph import quantitative_conditions
from .quantity_study import partitions, sha, write
from .train import metrics

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def representation(record):
    conditions, _ = quantitative_conditions(record)
    bits = np.zeros((4, 1024), dtype=np.float32)
    for component in record['components']:
        _, molecule = canonical(component['smiles'])
        bits[component['role']] = np.maximum(
            bits[component['role']], GENERATOR.GetFingerprintAsNumPy(molecule))
    return np.concatenate([bits.flatten(), conditions])


def estimator(config):
    cls = RandomForestRegressor if config['kind'] == 'forest' else ExtraTreesRegressor
    return cls(n_estimators=config['trees'], min_samples_leaf=config['leaf'],
               random_state=42, n_jobs=2)


def run(source='docs/quantity-v2', output='docs/morgan-v3',
        data='data/processed/reactions-quantity-v2.json'):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Choose a fresh output directory')
    if (source / 'ELN-results.json').exists():
        raise ValueError('Followup must be selected before ELN release')
    selection = json.loads((source / 'selection.json').read_text(encoding='utf-8'))
    plan = json.loads(Path('docs/morgan-plan.json').read_text(encoding='utf-8'))
    if sha(data) != selection['processed_sha256']:
        raise ValueError('Frozen processed dataset changed')
    dataset = json.loads(Path(data).read_text(encoding='utf-8'))
    development, _, folds = partitions(dataset)
    started = time.monotonic()
    x = np.stack([representation(record) for record in development])
    y = np.asarray([record['yield_percent'] for record in development])
    assignment = np.asarray([folds[record['group']] for record in development])
    write(output / 'run-protocol.json', {'plan_sha256': sha('docs/morgan-plan.json'),
          'source_selection_sha256': sha(source / 'selection.json'),
          'processed_sha256': sha(data), 'test_evaluated': False})
    baseline = next(row for row in selection['forest_trials']
                    if row['config'] == selection['selected_forest'])
    trials, failures = [], []
    for config in plan['candidates']:
        try:
            scores = []
            for number in range(3):
                train, validation = assignment != number, assignment == number
                model = estimator(config).fit(x[train], y[train])
                scores.append(metrics(y[validation], model.predict(x[validation]))['mae_percentage_points'])
            row = {'config': config, 'fold_validation_mae': scores,
                   'mean_validation_mae': float(np.mean(scores))}
            trials.append(row)
            print(json.dumps(row), flush=True)
        except Exception as error:
            failures.append({'config': config, 'error': type(error).__name__, 'reason': str(error)})
    eligible = [baseline] + [row for row in trials
        if row['mean_validation_mae'] <= baseline['mean_validation_mae'] - .5
        and sum(a < b for a, b in zip(row['fold_validation_mae'], baseline['fold_validation_mae'])) >= 2]
    chosen = min(eligible, key=lambda row: row['mean_validation_mae'])
    report = {'plan_sha256': sha('docs/morgan-plan.json'), 'implementation_sha256': sha(__file__),
              'baseline': baseline, 'trials': trials, 'failures': failures,
              'selected_forest': chosen['config'], 'runtime_seconds': time.monotonic() - started,
              'development_rows': len(development), 'test_evaluated': False}
    write(output / 'morgan-validation.json', report)
    selection['source_quantity_selection_sha256'] = sha(source / 'selection.json')
    selection['morgan_validation_sha256'] = sha(output / 'morgan-validation.json')
    selection['selected_forest'] = chosen['config']
    selection['feature_sha256']['morgan.py'] = sha(__file__)
    write(output / 'selection.json', selection)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='docs/quantity-v2')
    parser.add_argument('--output', default='docs/morgan-v3')
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
