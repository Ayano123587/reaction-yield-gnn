"""Selected CLI guards do not train or rescore observed cohorts."""
from unittest.mock import Mock
import pytest
from reaction_gnn import selected


def test_existing_training_directory_rejected_before_source_read(tmp_path, monkeypatch):
    fingerprint = Mock(side_effect=AssertionError('must not read'))
    monkeypatch.setattr(selected, 'sha', fingerprint)
    with pytest.raises(ValueError, match='Fresh'):
        selected.train('missing.json', tmp_path)
    fingerprint.assert_not_called()


def test_observed_evaluation_requires_ack_before_source_read(tmp_path, monkeypatch):
    fingerprint = Mock(side_effect=AssertionError('must not read'))
    monkeypatch.setattr(selected, 'sha', fingerprint)
    with pytest.raises(ValueError, match='explicit'):
        selected.evaluate('missing.json', 'missing-model', tmp_path/'result.json')
    fingerprint.assert_not_called()


def test_existing_observed_report_is_preserved(tmp_path):
    report = tmp_path/'result.json'
    report.write_text('preserve', encoding='utf-8')
    with pytest.raises(ValueError, match='Preserve'):
        selected.evaluate('missing', 'missing', report, acknowledge_observed=True)
    assert report.read_text() == 'preserve'


def test_frozen_configuration_matches_retained_model():
    import json
    from pathlib import Path
    config = json.loads(Path('docs/selected-model.json').read_text())
    assert config['epochs'] == 47
    assert config['gnn'] == {'name': 'legacy_role48', 'feature_version': 1, 'pooling': 'role', 'loss': 'mse'}
    assert config['training']['hidden'] == 48 and config['training']['layers'] == 3


def test_evaluation_requires_bound_training_receipt(tmp_path, monkeypatch):
    import json
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'processed_sha256':'bound'}),encoding='utf-8')
    monkeypatch.setattr(selected,'sha',lambda _:'bound')
    with pytest.raises(ValueError,match='training receipt'):
        selected.evaluate('fixture',tmp_path/'model',tmp_path/'report.json',acknowledge_observed=True,config_path=config)
