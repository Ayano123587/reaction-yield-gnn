"""Legacy scoring must be explicit and must not replace prior evidence."""
import importlib
from pathlib import Path
import pytest
from reaction_gnn.cli import main

RUNNERS = {'train': ('reaction_gnn.train', 'train'),
           'optimize': ('reaction_gnn.optimize', 'optimize'),
           'transfer-eval': ('reaction_gnn.transfer', 'evaluate_transfer')}


def mock_runner(monkeypatch, command, calls):
    module, name = RUNNERS[command]
    def runner(*args, **kwargs):
        calls.append((args, kwargs))
    monkeypatch.setattr(importlib.import_module(module), name, runner)


@pytest.mark.parametrize('command', RUNNERS)
def test_cli_no_acknowledgement_never_calls_scoring(monkeypatch, tmp_path, command):
    calls = []
    mock_runner(monkeypatch, command, calls)
    report, checkpoint = tmp_path/'report.json', tmp_path/'checkpoint.pt'
    with pytest.raises(SystemExit) as error:
        main([command, '--report', str(report), '--checkpoint', str(checkpoint)])
    assert error.value.code == 2 and calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('command', RUNNERS)
def test_cli_existing_evidence_is_unchanged(monkeypatch, tmp_path, command):
    calls = []
    mock_runner(monkeypatch, command, calls)
    report, checkpoint = tmp_path/'report.json', tmp_path/'checkpoint.pt'
    artifact = checkpoint if command == 'optimize' else report
    artifact.write_bytes(b'original scientific evidence')
    with pytest.raises(SystemExit) as error:
        main([command, '--report', str(report), '--checkpoint', str(checkpoint),
              '--acknowledge-observed-evaluation'])
    assert error.value.code == 2 and calls == []
    assert artifact.read_bytes() == b'original scientific evidence'
    assert sorted(tmp_path.iterdir()) == [artifact]


def test_cli_fresh_report_cannot_replace_optimization_protocol(monkeypatch, tmp_path):
    calls = []
    mock_runner(monkeypatch, 'optimize', calls)
    protocol = tmp_path/'selection-protocol.json'
    protocol.write_bytes(b'old frozen protocol')
    with pytest.raises(SystemExit):
        main(['optimize', '--report', str(tmp_path/'fresh.json'),
              '--checkpoint', str(tmp_path/'fresh.pt'), '--acknowledge-observed-evaluation'])
    assert calls == [] and protocol.read_bytes() == b'old frozen protocol'


@pytest.mark.parametrize('command', RUNNERS)
def test_cli_explicit_fresh_run_forwards_ack_without_real_training(monkeypatch, tmp_path, command):
    calls = []
    mock_runner(monkeypatch, command, calls)
    report, checkpoint = tmp_path/'report.json', tmp_path/'checkpoint.pt'
    if command == 'transfer-eval':
        checkpoint.write_bytes(b'existing input model is not an output')
    main([command, '--report', str(report), '--checkpoint', str(checkpoint),
          '--acknowledge-observed-evaluation'])
    assert len(calls) == 1 and calls[0][1]['acknowledge_observed'] is True
    assert not report.exists()


@pytest.mark.parametrize('command', RUNNERS)
def test_programmatic_api_also_refuses_implicit_or_existing_outputs(tmp_path, command):
    module, name = RUNNERS[command]
    runner = getattr(importlib.import_module(module), name)
    report, checkpoint = tmp_path/'report.json', tmp_path/'checkpoint.pt'
    args = (['missing.json', checkpoint, report] if command == 'train' else
            ['missing.json', report, checkpoint] if command == 'optimize' else
            ['missing.parquet', checkpoint, 'missing-development.json', report])
    with pytest.raises(ValueError, match='previously observed'):
        runner(*args)
    report.write_bytes(b'preserve me')
    with pytest.raises(ValueError, match='Preserve existing artifact'):
        runner(*args, acknowledge_observed=True)
    assert report.read_bytes() == b'preserve me'


def test_programmatic_split_sidecar_and_aliases_are_protected(tmp_path):
    from reaction_gnn.train import train
    from reaction_gnn.artifacts import observed_outputs
    report = tmp_path/'report.json'
    sidecar = tmp_path/'report-split-ids.json'
    sidecar.write_bytes(b'old split')
    with pytest.raises(ValueError, match='Preserve existing artifact'):
        train('missing.json', tmp_path/'new.pt', report, acknowledge_observed=True)
    assert sidecar.read_bytes() == b'old split'
    with pytest.raises(ValueError, match='distinct'):
        observed_outputs(True, 'ELN', [report, report])


def test_exclusive_report_write_preserves_file_created_after_preflight(tmp_path):
    from reaction_gnn.artifacts import observed_outputs,write_text_new
    report=tmp_path/'report.json'
    observed_outputs(True,'ELN',[report])
    report.write_bytes(b'another run completed first')
    with pytest.raises(FileExistsError):
        write_text_new(report,'replace it')
    assert report.read_bytes()==b'another run completed first'


def test_exclusive_checkpoint_write_never_serializes_over_existing_file(monkeypatch,tmp_path):
    import torch
    from reaction_gnn.artifacts import save_torch_new
    fresh=tmp_path/'fresh.pt'
    save_torch_new({'feature_version':1,'state_dict':{'weight':torch.tensor([.25])}},fresh)
    restored=torch.load(fresh,weights_only=True,map_location='cpu')
    assert restored['feature_version']==1
    torch.testing.assert_close(restored['state_dict']['weight'],torch.tensor([.25]))
    checkpoint=tmp_path/'model.pt'
    checkpoint.write_bytes(b'preserve model')
    def forbidden(*args,**kwargs):
        raise AssertionError('Serialization must not start for an existing checkpoint')
    monkeypatch.setattr(torch,'save',forbidden)
    with pytest.raises(FileExistsError):
        save_torch_new({},checkpoint)
    assert checkpoint.read_bytes()==b'preserve model'
