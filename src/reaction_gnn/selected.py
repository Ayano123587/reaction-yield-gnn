"""Fixed selected-model reproduction and explicit retrospective evaluation."""
import hashlib
import json
from pathlib import Path
from .artifacts import observed_outputs, write_text_new, save_torch_new


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(data, output, config_path='docs/selected-model.json'):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh output directory required')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if sha(data) != config['processed_sha256']:
        raise ValueError('Frozen processed dataset fingerprint mismatch')
    from .quantity_study import partitions, fit_graph, fit_forest
    from .graph import reaction_graph
    from .train import load_model, predict_record
    import torch
    import joblib
    records = json.loads(Path(data).read_text(encoding='utf-8'))
    development, _, _ = partitions(records)
    if len(development) != config['development_rows']:
        raise ValueError('Development cohort differs')
    torch.set_num_threads(config['training']['threads'])
    model, _, history = fit_graph([reaction_graph(r) for r in development], None,
                                 config['gnn'], config['training'], 42, epochs=config['epochs'])
    forest = fit_forest(development, config['forest'])
    output.mkdir(parents=True)
    state = {'state_dict': model.state_dict(), 'hidden': config['training']['hidden'],
             'layers': config['training']['layers'], 'pooling': config['gnn']['pooling'],
             'dropout': config['training']['dropout'], 'feature_version': 1}
    save_torch_new(state, output/'selected.pt')
    with (output/'forest.joblib').open('xb') as stream:
        joblib.dump(forest, stream)
    delta = abs(predict_record(model, development[0]) - predict_record(load_model(output/'selected.pt'), development[0]))
    if delta > 1e-5:
        raise AssertionError('Saved checkpoint replay differs')
    report = {'config': config, 'config_sha256': sha(config_path), 'training_rows': len(development),
              'training_history': history, 'test_evaluated': False, 'reload_delta_pp': delta,
              'checkpoint_sha256': sha(output/'selected.pt'), 'forest_sha256': sha(output/'forest.joblib')}
    write_text_new(output/'training.json', json.dumps(report, indent=2, allow_nan=False))
    return report


def evaluate(data, checkpoint_directory, report, *, acknowledge_observed=False, config_path='docs/selected-model.json'):
    observed_outputs(acknowledge_observed, 'ELN', [report])
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if sha(data) != config['processed_sha256']:
        raise ValueError('Frozen processed dataset fingerprint mismatch')
    checkpoint_directory = Path(checkpoint_directory)
    receipt_path = checkpoint_directory/'training.json'
    if not receipt_path.is_file():
        raise ValueError('Selected reproduction training receipt required; inference remains available for legacy checkpoints')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    expected = {'config_sha256': sha(config_path), 'checkpoint_sha256': sha(checkpoint_directory/'selected.pt'),
                'forest_sha256': sha(checkpoint_directory/'forest.joblib')}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint/configuration fingerprint mismatch')
    from .quantity_study import partitions, predictions, forest_predictions
    from .graph import reaction_graph
    from .train import load_model, metrics
    import joblib
    records = json.loads(Path(data).read_text(encoding='utf-8'))
    _, heldout, _ = partitions(records)
    if len(heldout) != config['observed_eln_rows']:
        raise ValueError('Observed cohort differs')
    checkpoint_directory = Path(checkpoint_directory)
    target = [r['yield_percent'] for r in heldout]
    graph = predictions(load_model(checkpoint_directory/'selected.pt'), [reaction_graph(r) for r in heldout], 1)
    forest = forest_predictions(joblib.load(checkpoint_directory/'forest.joblib'), heldout, config['forest'])
    result = {'observed_retrospective': True, 'rows': len(heldout),
              'gnn': metrics(target, graph), 'forest': metrics(target, forest),
              'checkpoint_sha256': sha(checkpoint_directory/'selected.pt'), 'config_sha256': sha(config_path)}
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    write_text_new(report, json.dumps(result, indent=2, allow_nan=False))
    return result
