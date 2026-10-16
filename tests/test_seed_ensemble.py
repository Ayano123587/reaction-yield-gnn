import pytest
from reaction_gnn.seed_ensemble import decide


def test_ensemble_gate_requires_gain_and_fold_support():
    assert decide([24,24,24],[23,23,24])['adopt_ensemble']
    assert not decide([24,24,24],[23.8,23.8,23.8])['adopt_ensemble']
    assert not decide([24,24,24],[20,24.1,24.1])['adopt_ensemble']


@pytest.mark.parametrize('value',[[1,2],[1,float('nan'),2],[1,-1,2]])
def test_ensemble_invalid_fold_errors(value):
    with pytest.raises(ValueError):decide(value,[1,2,3])
