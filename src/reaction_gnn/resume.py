"""Hash-bound complete-line prediction snapshots for compute migration."""
import hashlib
import json
import math
from pathlib import Path


def snapshot(source,destination):
    source,destination=Path(source),Path(destination)
    with source.open('rb') as stream:
        size=source.stat().st_size
        content=stream.read(size)
    end=content.rfind(b'\n')+1
    if not end:raise ValueError('No complete prediction lines')
    content=content[:end]
    rows=decode(content)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:stream.write(content)
    return {'sha256':hashlib.sha256(content).hexdigest(),'rows':len(rows),
            'bytes':len(content),'captured_source_bytes':size,
            'partial_tail_bytes_excluded':size-end}


def decode(content):
    rows=[];seen=set()
    if not content.endswith(b'\n'):raise ValueError('Snapshot must end on complete line')
    for line in content.decode('utf-8').splitlines():
        row=json.loads(line)
        if set(row)!={'reaction_id','target','gnn','forest'}:raise ValueError('Unexpected prediction schema')
        identifier=row['reaction_id']
        if not isinstance(identifier,str) or not identifier or identifier in seen:raise ValueError('Invalid/duplicate prefix identifier')
        for key in ['target','gnn','forest']:
            value=row[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=100:
                raise ValueError('Nonfinite/out-of-range prefix values')
        seen.add(identifier);rows.append(row)
    if not rows:raise ValueError('Empty prefix')
    return rows


def load(path,manifest):
    content=Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest()!=manifest['sha256'] or len(content)!=manifest['bytes']:
        raise ValueError('Snapshot bytes changed')
    rows=decode(content)
    if len(rows)!=manifest['rows']:raise ValueError('Snapshot count changed')
    return content.decode('utf-8'),rows


def verify_record(record,prediction):
    if record['reaction_id']!=prediction['reaction_id'] or record['yield_percent']!=prediction['target']:
        raise ValueError('Snapshot source order or target changed')
