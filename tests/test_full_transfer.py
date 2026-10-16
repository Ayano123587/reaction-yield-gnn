from pathlib import Path
import pytest
from reaction_gnn.full_transfer import evaluate


def test_full_source_requires_ack_before_loading(tmp_path):
    with pytest.raises(ValueError, match='acknowledge'):
        evaluate('missing.parquet', tmp_path/'report.json', tmp_path/'predictions.jsonl')
    assert list(tmp_path.iterdir()) == []


def test_full_source_existing_output_preserved(tmp_path):
    output=tmp_path/'report.json';output.write_text('original')
    with pytest.raises(ValueError):
        evaluate('missing.parquet',output,tmp_path/'private.jsonl',True)
    assert output.read_text()=='original'


def test_full_source_private_rows_cannot_escape_data(tmp_path):
    with pytest.raises(ValueError):
        evaluate('missing.parquet',tmp_path/'report.json',tmp_path/'private.jsonl',True)
    assert list(tmp_path.iterdir())==[]
