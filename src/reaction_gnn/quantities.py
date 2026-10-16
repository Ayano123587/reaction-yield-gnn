"""Input-only, unit-safe reported amounts; no outcome-derived quantities."""
from collections import Counter
import hashlib,json,math
from pathlib import Path
import pyarrow.parquet as pq
from ord_schema.proto import reaction_pb2 as pb
from .data import extract_inputs,extract_target,canonical,scaffold_group,ROLES,EXPECTED_SHA256,input_signature

MOLES={pb.Moles.MOLE:1.,pb.Moles.MILLIMOLE:1e-3,pb.Moles.MICROMOLE:1e-6,pb.Moles.NANOMOLE:1e-9}
VOLUME={pb.Volume.LITER:1.,pb.Volume.MILLILITER:1e-3,pb.Volume.MICROLITER:1e-6,pb.Volume.NANOLITER:1e-9}


def amount(component,field,units):
    if not component.HasField('amount') or not component.amount.HasField(field):return None
    quantity=getattr(component.amount,field)
    if not quantity.HasField('value'):return None
    value=float(quantity.value)
    if not math.isfinite(value) or value<0:raise ValueError('Input amounts must be finite and nonnegative')
    if quantity.units not in units:return None
    result=value*units[quantity.units]
    if not math.isfinite(result):raise ValueError('Converted input amount is nonfinite')
    return result


def extract_quantitative_inputs(reaction):
    base=extract_inputs(reaction);components=[]
    for key in sorted(reaction.inputs):
        for component in reaction.inputs[key].components:
            if component.reaction_role not in ROLES:continue
            identifiers=[i.value for i in component.identifiers if i.type==pb.CompoundIdentifier.SMILES]
            if not identifiers:continue  # Legacy allow-list already verified only valid sentinels may be skipped.
            smi,_=canonical(identifiers[0])
            components.append({'smiles':smi,'role':ROLES[component.reaction_role],
                'moles':amount(component,'moles',MOLES),'volume_l':amount(component,'volume',VOLUME),
                'is_limiting':bool(component.is_limiting)})
    if [(c['smiles'],c['role']) for c in components]!=[(c['smiles'],c['role']) for c in base['components']]:raise ValueError('Quantity/component alignment changed')
    return {'components':components,'temperature_c':base['temperature_c']}


def prepare_quantitative(path,output,original='data/processed/reactions.json'):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA256:raise ValueError('Pinned ELN source digest changed')
    table=pq.read_table(path,columns=['reaction_id','reaction']);records=[];dropped=Counter()
    for item in table.to_pylist():
        reaction=pb.Reaction();reaction.ParseFromString(item['reaction'])
        try:
            inputs=extract_quantitative_inputs(reaction)
            records.append({'reaction_id':item['reaction_id'],**inputs,'yield_percent':extract_target(reaction),'group':scaffold_group(inputs['components'])})
        except ValueError as error:dropped[str(error)]+=1
    old=json.loads(Path(original).read_text(encoding='utf-8'))
    if [(r['reaction_id'],r['group'],input_signature(r)) for r in records]!=[(r['reaction_id'],r['group'],input_signature(r)) for r in old['records']]:raise ValueError('ELN cohort/groups/input signatures changed; stop before training')
    if [r['yield_percent'] for r in records]!=[r['yield_percent'] for r in old['records']]:raise ValueError('ELN labels changed; stop before training')
    result={'dataset_id':old['dataset_id'],'records':records,'raw_rows':len(table),'dropped':dict(dropped),'feature_version':2,'feature_contract':'input SMILES/roles, temperature setpoint, reported moles/volumes and limiting flag only','raw_sha256':EXPECTED_SHA256}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return {'records':len(records),'raw_rows':len(table),'dropped':dict(dropped),'processed_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_ids_groups_signatures_unchanged':True}
