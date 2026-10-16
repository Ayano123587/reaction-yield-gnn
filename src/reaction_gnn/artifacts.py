"""Fail before evaluation or fitting when an observed run is not explicit/fresh."""
from pathlib import Path


def observed_outputs(acknowledge_observed, cohort, outputs):
    if acknowledge_observed is not True:
        raise ValueError(f'This legacy operation scores the previously observed {cohort} cohort; '
                         'explicit acknowledge_observed=True is required. '
                         'Use root train.py for validation-only training.')
    paths = [Path(path) for path in outputs]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError('Output artifact paths must be distinct')
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ValueError(f'Preserve existing artifact: {path}; choose fresh output paths')
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise ValueError(f'Output ancestor is not a directory: {parent}')


def train_outputs(report, checkpoint):
    report = Path(report)
    return [report, checkpoint, report.with_name(report.stem + '-split-ids.json')]


def optimize_outputs(report, checkpoint):
    report = Path(report)
    return [report, checkpoint, report.with_name('selection-protocol.json'),
            report.with_name('selection-progress.jsonl')]


def write_text_new(path, text):
    """Exclusive creation also protects an artifact appearing after preflight."""
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def save_torch_new(state, path):
    import torch
    with Path(path).open('xb') as stream:
        torch.save(state, stream)
