"""Prespecified graph-loss comparison on frozen scaffold development folds."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .data import audit_split_integrity
from .graph import reaction_graph as old_graph
from .quantity_graph import reaction_graph as quantity_graph
from .quantity_study import partitions, fit_graph, predictions, save_model, sha, write
from .train import metrics


def select_loss(baseline, trials):
    eligible = [baseline] + [row for row in trials
        if row['mean_validation_mae'] <= baseline['mean_validation_mae'] - .5
        and sum(a < b for a, b in zip(row['fold_validation_mae'], baseline['fold_validation_mae'])) >= 2]
    return min(eligible, key=lambda row: row['mean_validation_mae'])


def run(source='docs/morgan-v3', output='docs/loss-v4'):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Preserve history; choose a fresh output directory')
    if (source / 'ELN-results.json').exists():
        raise ValueError('Loss selection must precede ELN release')
    selection = json.loads((source / 'selection.json').read_text(encoding='utf-8'))
    plan = json.loads(Path('docs/loss-plan.json').read_text(encoding='utf-8'))
    fixed = json.loads(Path('docs/quantity-plan.json').read_text(encoding='utf-8'))['fixed_training']
    data = Path('data/processed/reactions-quantity-v2.json')
    if sha(data) != selection['processed_sha256']:
        raise ValueError('Frozen development data changed')
    development, _, folds = partitions(json.loads(data.read_text(encoding='utf-8')))
    torch.set_num_threads(2)
    started = time.monotonic()
    graphs = {(version, row['reaction_id']): (old_graph(row) if version == 1 else quantity_graph(row))
              for version in [1, 2] for row in development}
    baseline = next(row for row in selection['gnn_trials'] if row['config']['feature_version'] == 1)
    write(output / 'run-protocol.json', {'plan_sha256': sha('docs/loss-plan.json'),
          'source_selection_sha256': sha(source / 'selection.json'), 'test_evaluated': False})
    trials, failures = [], []
    local = Path('data/interim') / output.name
    for config in plan['candidates']:
        try:
            scores, epochs, histories = [], [], []
            version = config['feature_version']
            for number in range(3):
                train = [row for row in development if folds[row['group']] != number]
                validation = [row for row in development if folds[row['group']] == number]
                audit_split_integrity({'train': train, 'validation': validation})
                train_graphs = [graphs[version, row['reaction_id']] for row in train]
                validation_graphs = [graphs[version, row['reaction_id']] for row in validation]
                target = np.asarray([row['yield_percent'] for row in validation])
                model, epoch, history = fit_graph(train_graphs, (validation_graphs, target),
                                                  config, fixed, 42 + number)
                predicted = predictions(model, validation_graphs, version)
                scores.append(metrics(target, predicted)['mae_percentage_points'])
                epochs.append(epoch)
                histories.append(history)
                save_model(model, config, fixed, local / f'{config["name"]}-fold{number}.pt')
                write(local / f'{config["name"]}-fold{number}-predictions.json',
                      [{'reaction_id': row['reaction_id'], 'target': float(y), 'prediction': float(p)}
                       for row, y, p in zip(validation, target, predicted)])
                print(json.dumps({'config': config['name'], 'fold': number,
                                  'mae': scores[-1], 'best_epoch': epoch}), flush=True)
            trials.append({'config': config, 'fold_validation_mae': scores,
                           'mean_validation_mae': float(np.mean(scores)),
                           'best_epochs': epochs, 'history': histories})
        except Exception as error:
            failures.append({'config': config, 'error': type(error).__name__, 'reason': str(error)})
    # select_gnn identifies baselines by feature_version, so apply the fixed gate
    # directly: robust legacy candidates share feature_version=1 with the MSE baseline.
    chosen = select_loss(baseline, trials)
    result = {'plan_sha256': sha('docs/loss-plan.json'), 'implementation_sha256': sha(__file__),
              'baseline': baseline, 'trials': trials, 'failures': failures,
              'selected_gnn': chosen['config'], 'runtime_seconds': time.monotonic() - started,
              'test_evaluated': False}
    write(output / 'loss-validation.json', result)
    selection['source_morgan_selection_sha256'] = sha(source / 'selection.json')
    selection['loss_validation_sha256'] = sha(output / 'loss-validation.json')
    selection['selected_gnn'] = chosen['config']
    selection['final_refit_epochs'] = max(1, round(float(np.median(chosen['best_epochs']))))
    write(output / 'selection.json', selection)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
