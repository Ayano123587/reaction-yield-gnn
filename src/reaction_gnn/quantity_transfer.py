"""Evaluate frozen ELN models on the independent, hash-selected ORD HTE source."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq
import torch
from ord_schema.proto import reaction_pb2 as pb
from .data import extract_target, input_signature, scaffold_group
from .graph import reaction_graph as old_graph
from .quantities import extract_quantitative_inputs
from .quantity_graph import reaction_graph as quantity_graph
from .quantity_study import sha, write, predictions, forest_predictions, partitions
from .train import load_model, metrics
from .transfer import TRANSFER_ID, TRANSFER_SHA


def evaluate(path, selection_directory, data='data/processed/reactions-quantity-v2.json'):
    directory = Path(selection_directory)
    destination = directory / 'HTE-transfer-results.json'
    if destination.exists():
        raise ValueError('Preserve the existing independent-source evaluation')
    selection = json.loads((directory / 'selection.json').read_text(encoding='utf-8'))
    eln_result = json.loads((directory / 'ELN-results.json').read_text(encoding='utf-8'))
    if sha(data) != selection['processed_sha256'] or sha(path) != TRANSFER_SHA:
        raise ValueError('Frozen source data changed')
    if sha(directory / 'selection.json') != eln_result['selection_sha256']:
        raise ValueError('ELN selection changed')
    for name, digest in selection['feature_sha256'].items():
        if sha(Path('src/reaction_gnn') / name) != digest:
            raise ValueError('Selected representation changed')
    model_directory = Path('models') / directory.name
    if sha(model_directory / 'selected.pt') != eln_result['gnn_checkpoint_sha256']:
        raise ValueError('Selected graph checkpoint changed')
    if sha(model_directory / 'forest.joblib') != eln_result['forest_checkpoint_sha256']:
        raise ValueError('Selected forest checkpoint changed')
    dataset = json.loads(Path(data).read_text(encoding='utf-8'))
    development, _, _ = partitions(dataset)
    seen = {input_signature(record) for record in dataset['records']}
    items = sorted(pq.read_table(path, columns=['reaction_id', 'reaction']).to_pylist(),
                   key=lambda row: hashlib.sha256(row['reaction_id'].encode()).hexdigest())[:2048]
    records, dropped = [], Counter()
    for item in items:
        reaction = pb.Reaction()
        reaction.ParseFromString(item['reaction'])
        try:
            record = {'reaction_id': item['reaction_id'], **extract_quantitative_inputs(reaction)}
            if input_signature(record) in seen:
                raise ValueError('Exact structural/temperature input clone from ELN cohort')
            record['yield_percent'] = extract_target(reaction)
            record['group'] = scaffold_group(record['components'])
            records.append(record)
        except ValueError as error:
            dropped[str(error)] += 1
    if not records:
        raise ValueError('No valid independent-source rows')
    torch.set_num_threads(2)
    model = load_model(model_directory / 'selected.pt')
    version = selection['selected_gnn']['feature_version']
    graph = old_graph if version == 1 else quantity_graph
    gnn = predictions(model, [graph(record) for record in records], version)
    forest = forest_predictions(joblib.load(model_directory / 'forest.joblib'), records,
                                selection['selected_forest'])
    target = np.asarray([record['yield_percent'] for record in records])
    median = float(np.median([record['yield_percent'] for record in development]))
    write(Path('data/interim') / directory.name / 'HTE-transfer-predictions.json', [
        {'reaction_id': record['reaction_id'], 'target': float(y), 'gnn': float(g), 'forest': float(f)}
        for record, y, g, f in zip(records, target, gnn, forest)])
    result = {
        'dataset_id': TRANSFER_ID, 'source_sha256': TRANSFER_SHA,
        'selection_sha256': sha(directory / 'selection.json'),
        'gnn_checkpoint_sha256': eln_result['gnn_checkpoint_sha256'],
        'forest_checkpoint_sha256': eln_result['forest_checkpoint_sha256'],
        'sampling': 'Lowest SHA256(reaction_id), 2048 identifiers selected before labels',
        'candidate_rows': len(items), 'retained_rows': len(records), 'dropped': dict(dropped),
        'gnn': metrics(target, gnn), 'selected_forest': metrics(target, forest),
        'eln_development_median': metrics(target, np.full(len(target), median)),
        'principal_scaffolds': len({r['group'] for r in records}),
        'principal_scaffolds_overlapping_eln': len({r['group'] for r in records} &
                                                   {r['group'] for r in dataset['records']}),
        'interpretation': 'Independent-source assay-yield transfer, without tuning or adaptation. '
                          'Narrow two-amine/four-aryl-halide chemistry; scaffold overlap is allowed '
                          'and reported. Assay yield differs from ELN measurements. No broad '
                          'chemistry generalization or calibrated uncertainty claim.'}
    write(destination, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/external/cn-50688.parquet')
    parser.add_argument('--selection-directory', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source, args.selection_directory), indent=2))
