"""Complete-source frozen HTE evaluation; no training or capped sampling."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from ord_schema.proto import reaction_pb2 as pb
import pyarrow.parquet as pq
import torch

from .artifacts import observed_outputs, write_text_new
from .data import extract_target, input_signature, scaffold_group
from .graph import reaction_graph as legacy_graph
from .quantities import extract_quantitative_inputs
from .quantity_graph import reaction_graph as quantitative_graph
from .quantity_study import sha, partitions, predictions, forest_predictions
from .train import load_model, metrics
from .resume import load as load_snapshot, verify_record


def evaluate(source, output, private_output, acknowledge_observed=False,
             plan_path='docs/full-hte-plan.json', resume_manifest=None, prefix_path=None):
    """Score every eligible row with immutable checkpoints; private JSONL only."""
    output, private_output = Path(output), Path(private_output)
    observed_outputs(acknowledge_observed, 'Full HTE (includes observed subset)',
                     [output, private_output])
    # Raw source/predictions must never escape the ignored data tree.
    private_output.resolve().relative_to(Path('data').resolve())
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    directory = Path(plan['selection_directory'])
    selection = json.loads((directory/'selection.json').read_text(encoding='utf-8'))
    eln = json.loads((directory/'ELN-results.json').read_text(encoding='utf-8'))
    data = Path('data/processed/reactions-quantity-v2.json')
    model_directory = Path('models')/directory.name
    for path, expected in [(source, plan['source_sha256']),
                           (data, plan['processed_eln_sha256']),
                           (model_directory/'selected.pt', plan['gnn_checkpoint_sha256']),
                           (model_directory/'forest.joblib', plan['forest_checkpoint_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Frozen artifact changed: {path}')
    if sha(directory/'selection.json') != eln['selection_sha256']:
        raise ValueError('Selected configuration changed')
    for name, digest in selection['feature_sha256'].items():
        if sha(Path('src/reaction_gnn')/name) != digest:
            raise ValueError('Selected graph features changed')
    table = pq.read_table(source, columns=['reaction_id','reaction'])
    if len(table) != plan['source_rows_expected']:
        raise ValueError('Unexpected source count')
    identifiers = table.column('reaction_id').to_pylist()
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate source reaction identifiers')
    observed = set(sorted(identifiers, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:2048])
    dataset = json.loads(data.read_text(encoding='utf-8'))
    development, _, _ = partitions(dataset)
    seen = {input_signature(r) for r in dataset['records']}
    eln_groups = {r['group'] for r in dataset['records']}
    median = float(np.median([r['yield_percent'] for r in development]))
    torch.set_num_threads(plan['threads'])
    model = load_model(model_directory/'selected.pt')
    forest = joblib.load(model_directory/'forest.joblib')
    version = selection['selected_gnn']['feature_version']
    graph = legacy_graph if version == 1 else quantitative_graph
    prefix_text, prefix_rows = '', []
    if resume_manifest is not None:
        for key in ['source_sha256','gnn_checkpoint_sha256','forest_checkpoint_sha256','processed_eln_sha256']:
            if resume_manifest[key] != plan[key]:raise ValueError('Resume artifact identity changed')
        if resume_manifest['plan_sha256'] != sha(plan_path):raise ValueError('Resume policy changed')
        prefix_text,prefix_rows=load_snapshot(prefix_path,resume_manifest['prefix'])
    prefix_index=0
    target, gnn_values, forest_values = [], [], []
    dropped, groups = Counter(), set()
    retained_observed = 0
    started = time.monotonic()
    private_output.parent.mkdir(parents=True, exist_ok=True)
    with private_output.open('x', encoding='utf-8', newline='') as private:
        private.write(prefix_text)
        processed = 0
        for batch in table.to_batches(max_chunksize=plan['batch_rows']):
            processed += len(batch)
            if processed % 4096 == 0 or processed == len(table):
                print(f'Processing full source: {processed}/{len(table)} rows', flush=True)
            records = []
            for item in batch.to_pylist():
                reaction = pb.Reaction()
                reaction.ParseFromString(item['reaction'])
                try:
                    r = {'reaction_id': item['reaction_id'], **extract_quantitative_inputs(reaction)}
                    if input_signature(r) in seen:
                        raise ValueError('Exact structural/temperature input clone from ELN cohort')
                    r['yield_percent'] = extract_target(reaction)
                    r['group'] = scaffold_group(r['components'])
                    records.append(r)
                except ValueError as error:
                    dropped[str(error)] += 1
            remaining=[]
            for r in records:
                if prefix_index<len(prefix_rows):
                    cached=prefix_rows[prefix_index];verify_record(r,cached);prefix_index+=1
                    target.append(cached['target']);gnn_values.append(cached['gnn']);forest_values.append(cached['forest'])
                    retained_observed += r['reaction_id'] in observed;groups.add(r['group'])
                else:remaining.append(r)
            records=remaining
            if not records:
                continue
            g = predictions(model, [graph(r) for r in records], version)
            f = forest_predictions(forest, records, selection['selected_forest'])
            if not np.isfinite(f).all():
                raise ValueError('Nonfinite forest predictions')
            for r, gv, fv in zip(records, g, f):
                y = r['yield_percent']
                target.append(y); gnn_values.append(float(gv)); forest_values.append(float(fv))
                retained_observed += r['reaction_id'] in observed
                groups.add(r['group'])
                private.write(json.dumps({'reaction_id':r['reaction_id'],'target':y,
                                          'gnn':float(gv),'forest':float(fv)},allow_nan=False)+'\n')
    if prefix_index!=len(prefix_rows):raise ValueError('Source ended before snapshot prefix')
    if not target or len(target)+sum(dropped.values()) != len(table):
        raise ValueError('Full source accounting failed')
    result = {'scope':plan['scope'],'test_exposure':plan['test_exposure'],
              'plan_sha256':sha(plan_path),'implementation_sha256':sha(__file__),
              'resume_manifest':resume_manifest,'prefix_rows_reused_without_model_inference':prefix_index,
              'source_sha256':sha(source),'source_rows':len(table),
              'retained_rows':len(target),'dropped':dict(dropped),
              'previously_observed_identifier_count':len(observed),
              'previously_observed_retained_rows':retained_observed,
              'newly_evaluated_retained_rows':len(target)-retained_observed,
              'selection_sha256':sha(directory/'selection.json'),
              'gnn_checkpoint_sha256':sha(model_directory/'selected.pt'),
              'forest_checkpoint_sha256':sha(model_directory/'forest.joblib'),
              'gnn':metrics(target,gnn_values),'selected_forest':metrics(target,forest_values),
              'eln_development_median':metrics(target,np.full(len(target),median)),
              'principal_scaffolds':len(groups),'principal_scaffolds_overlapping_eln':len(groups&eln_groups),
              'private_predictions_sha256':sha(private_output),
              'runtime_seconds':time.monotonic()-started,
              'environment':{'torch':torch.__version__,'device':'cpu','threads':plan['threads']}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_new(output,json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='data/external/cn-50688.parquet')
    p.add_argument('--output',required=True)
    p.add_argument('--private-output',required=True)
    p.add_argument('--acknowledge-observed-subset',action='store_true')
    p.add_argument('--resume-manifest',required=True)
    p.add_argument('--prefix',required=True)
    a=p.parse_args(argv)
    print(json.dumps(evaluate(a.source,a.output,a.private_output,a.acknowledge_observed_subset,
            resume_manifest=json.loads(Path(a.resume_manifest).read_text(encoding='utf-8')),prefix_path=a.prefix),indent=2))

if __name__=='__main__':
    main()
