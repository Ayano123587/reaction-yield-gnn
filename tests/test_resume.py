import json
import pytest
from reaction_gnn.resume import snapshot,load,decode,verify_record


def line(identifier='a',target=20):
    return (json.dumps({'reaction_id':identifier,'target':target,'gnn':30,'forest':40})+'\n').encode()


def test_snapshot_complete_prefix_immutable_after_append(tmp_path):
    source=tmp_path/'live';source.write_bytes(line()+b'{"unfinished":')
    destination=tmp_path/'prefix';manifest=snapshot(source,destination)
    source.write_bytes(line()+line('b'))
    text,rows=load(destination,manifest)
    assert len(rows)==1 and text.encode()==line()
    assert manifest['partial_tail_bytes_excluded']>0
    with pytest.raises(FileExistsError):snapshot(source,destination)


def test_hash_and_order_target_guards(tmp_path):
    source=tmp_path/'live';source.write_bytes(line());destination=tmp_path/'prefix'
    manifest=snapshot(source,destination)
    verify_record({'reaction_id':'a','yield_percent':20},load(destination,manifest)[1][0])
    for record in [{'reaction_id':'b','yield_percent':20},{'reaction_id':'a','yield_percent':21}]:
        with pytest.raises(ValueError):verify_record(record,load(destination,manifest)[1][0])
    destination.write_bytes(line('b'))
    with pytest.raises(ValueError):load(destination,manifest)


@pytest.mark.parametrize('content',[line()+line(),line(target=float('nan')),line(target=-1),line()[:-1],b''])
def test_invalid_snapshot(content):
    with pytest.raises(ValueError):decode(content)
