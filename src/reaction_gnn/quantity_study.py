"""Validation-only quantity GNN study and explicit frozen-model ELN evaluation."""
from copy import deepcopy
import argparse,hashlib,json,platform,time
from importlib.metadata import version as package_version
from pathlib import Path
import joblib
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from .data import split_records,audit_split_integrity
from .graph import reaction_graph as old_graph
from .quantity_graph import reaction_graph as quantity_graph,quantitative_conditions
from .quantity_model import QuantityGNN
from .model import YieldGNN
from .train import metrics,fingerprint,load_model,predict_record


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def loader(graphs,version,batch_size=32,shuffle=False,seed=42):
    return DataLoader(graphs,batch_size=batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),follow_batch=['component_roles'] if version==2 else [])


def predictions(model,graphs,version):
    model.eval()
    with torch.no_grad():values=torch.cat([model(batch) for batch in loader(graphs,version)]).numpy()*100
    if not np.isfinite(values).all() or (values<0).any() or (values>100).any():raise ValueError('Invalid model forecasts')
    return values


def model_for(config,fixed):
    cls=YieldGNN if config['feature_version']==1 else QuantityGNN
    return cls(fixed['hidden'],fixed['layers'],config['pooling'],fixed['dropout'])


def fit_graph(graphs,validation,config,fixed,seed,epochs=None):
    torch.manual_seed(seed);model=model_for(config,fixed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=fixed['lr'],weight_decay=fixed['weight_decay'])
    batches=loader(graphs,config['feature_version'],fixed['batch_size'],True,seed)
    history=[];best=float('inf');state=None;best_epoch=0
    maximum=epochs if epochs is not None else fixed['max_epochs']
    for epoch in range(maximum):
        model.train();total=0.;maximum_gradient=0.
        for batch in batches:
            output=model(batch)
            loss=torch.nn.functional.mse_loss(output,batch.y) if config['loss']=='mse' else torch.nn.functional.smooth_l1_loss(output,batch.y,beta=config['beta'])
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            maximum_gradient=max(maximum_gradient,float(norm));optimizer.step();total+=float(loss.detach())*batch.num_graphs
        item={'epoch':epoch+1,'train_loss_fraction':total/len(graphs),'maximum_gradient_norm_before_clip':maximum_gradient}
        if validation is not None:
            graph_val,target=validation;score=metrics(target,predictions(model,graph_val,config['feature_version']))['mae_percentage_points'];item['validation_mae']=score
            if score<best:best=score;state=deepcopy(model.state_dict());best_epoch=epoch+1
        history.append(item)
        if validation is not None and epoch+1>=fixed['minimum_epochs'] and epoch+1-best_epoch>=fixed['early_stopping_patience']:break
    if validation is not None:
        if state is None:raise ValueError('No finite validation checkpoint')
        model.load_state_dict(state)
    return model,best_epoch if validation is not None else maximum,history


def save_model(model,config,fixed,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'hidden':fixed['hidden'],'layers':fixed['layers'],'pooling':config['pooling'],'dropout':fixed['dropout'],'feature_version':config['feature_version']},path)


def partitions(dataset):
    splits=split_records(dataset['records'],42);audit_split_integrity(splits)
    dev=splits['train']+splits['validation'];groups=sorted({r['group'] for r in dev});np.random.default_rng(42).shuffle(groups)
    fold={g:i%3 for i,g in enumerate(groups)}
    return dev,splits['test'],fold


def augmented_fingerprint(record):return np.concatenate([fingerprint(record)[:-2],quantitative_conditions(record)[0]])


def forest_configurations():
    return [{'name':f'{kind}_{version}_leaf{leaf}','kind':kind,'feature_version':version,'leaf':leaf,'trees':200 if version==1 else 400} for kind,version in [('forest',1),('forest',2),('extra',2)] for leaf in [1,3,8]]


def fit_forest(records,config):
    x=np.stack([(fingerprint(r) if config['feature_version']==1 else augmented_fingerprint(r)) for r in records]);y=np.asarray([r['yield_percent'] for r in records])
    cls=RandomForestRegressor if config['kind']=='forest' else ExtraTreesRegressor
    return cls(n_estimators=config['trees'],min_samples_leaf=config['leaf'],random_state=42,n_jobs=2).fit(x,y)


def forest_predictions(model,records,config):
    return model.predict(np.stack([(fingerprint(r) if config['feature_version']==1 else augmented_fingerprint(r)) for r in records]))


def select_gnn(trials):
    baseline=next(t for t in trials if t['config']['feature_version']==1)
    eligible=[baseline]
    for t in trials:
        if t is baseline:continue
        wins=sum(x<y for x,y in zip(t['fold_validation_mae'],baseline['fold_validation_mae']))
        if t['mean_validation_mae']<=baseline['mean_validation_mae']-.5 and wins>=2:eligible.append(t)
    return min(eligible,key=lambda t:t['mean_validation_mae'])


def study(data='data/processed/reactions-quantity-v2.json',output='docs/quantity-v2',plan_path='docs/quantity-plan.json'):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Preserve existing quantity study; choose a fresh output directory')
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'));dataset=json.loads(Path(data).read_text(encoding='utf-8'));fixed=plan['fixed_training']
    if dataset['feature_version']!=2 or dataset['raw_sha256']!=plan['raw_eln_sha256']:raise ValueError('Prepared dataset/source contract changed')
    torch.set_num_threads(fixed['threads']);started=time.monotonic();dev,_,fold=partitions(dataset)
    graphs={(version,r['reaction_id']):(old_graph(r) if version==1 else quantity_graph(r)) for version in [1,2] for r in dev}
    local=Path('data/interim')/output.name;local.mkdir(parents=True,exist_ok=True);trials=[];failures=[]
    write(output/'run-protocol.json',{'plan_sha256':sha(plan_path),'processed_sha256':sha(data),'implementation_sha256':sha(__file__),'fold_assignment_sha256':hashlib.sha256(json.dumps(sorted(fold.items())).encode()).hexdigest(),'development_rows':len(dev),'test_evaluated':False})
    for config in plan['configs']:
        scores=[];epochs=[];histories=[]
        try:
            for number in range(3):
                train=[r for r in dev if fold[r['group']]!=number];val=[r for r in dev if fold[r['group']]==number];audit_split_integrity({'train':train,'validation':val})
                version=config['feature_version'];train_graph=[graphs[version,r['reaction_id']] for r in train];val_graph=[graphs[version,r['reaction_id']] for r in val];target=np.asarray([r['yield_percent'] for r in val])
                model,epoch,history=fit_graph(train_graph,(val_graph,target),config,fixed,42+number)
                prediction=predictions(model,val_graph,version);score=metrics(target,prediction)['mae_percentage_points'];scores.append(score);epochs.append(epoch);histories.append(history)
                write(local/(config['name']+f'-fold{number}-predictions.json'),[{'reaction_id':r['reaction_id'],'target':float(y),'prediction':float(p)} for r,y,p in zip(val,target,prediction)])
                save_model(model,config,fixed,local/(config['name']+f'-fold{number}.pt'))
                progress={'config':config['name'],'fold':number,'validation_mae':score,'best_epoch':epoch,'executed_epochs':len(history)}
                print(json.dumps(progress),flush=True)
                output.mkdir(parents=True,exist_ok=True)
                with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(progress)+'\n')
            trials.append({'config':config,'fold_validation_mae':scores,'mean_validation_mae':float(np.mean(scores)),'best_epochs':epochs,'history':histories})
        except Exception as error:
            failures.append({'config':config['name'],'error':type(error).__name__,'reason':str(error)});print(json.dumps(failures[-1]),flush=True)
    if not any(t['config']['feature_version']==1 for t in trials):raise ValueError('Required legacy comparison failed')
    write(output/'gnn-validation.json',{'trials':trials,'failures':failures})
    chosen=select_gnn(trials);forest_trials=[]
    for config in forest_configurations():
        scores=[]
        for number in range(3):
            train=[r for r in dev if fold[r['group']]!=number];val=[r for r in dev if fold[r['group']]==number]
            model=fit_forest(train,config);scores.append(metrics([r['yield_percent'] for r in val],forest_predictions(model,val,config))['mae_percentage_points'])
        forest_trials.append({'config':config,'fold_validation_mae':scores,'mean_validation_mae':float(np.mean(scores))})
    selected_forest=min(forest_trials,key=lambda t:t['mean_validation_mae'])
    result={'plan_sha256':sha(plan_path),'processed_sha256':sha(data),'implementation_sha256':sha(__file__),'feature_sha256':{p:sha('src/reaction_gnn/'+p) for p in ['quantities.py','quantity_graph.py','quantity_model.py']},'gnn_trials':trials,'forest_trials':forest_trials,'failures':failures,'selected_gnn':chosen['config'],'selected_forest':selected_forest['config'],'final_refit_epochs':max(1,round(float(np.median(chosen['best_epochs'])))),'development_rows':len(dev),'test_evaluated':False,'runtime_seconds':time.monotonic()-started,'environment':{'python':platform.python_version(),'torch':torch.__version__,'threads':fixed['threads'],'device':'cpu','packages':{name:package_version(name) for name in ['torch-geometric','rdkit','numpy','scikit-learn','ord-schema','pyarrow']}},'disclosure':plan['disclosure']}
    write(output/'selection.json',result);return result


def smoke(data='data/processed/reactions-quantity-v2.json',output='data/interim/quantity-smoke',plan_path='docs/quantity-plan.json'):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Preserve existing smoke run')
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'));dataset=json.loads(Path(data).read_text(encoding='utf-8'));dev,_,fold=partitions(dataset)
    config=plan['configs'][-1];fixed=plan['fixed_training'];torch.set_num_threads(2)
    train=[r for r in dev if fold[r['group']]!=0][:96];val=[r for r in dev if fold[r['group']]==0][:32];audit_split_integrity({'train':train,'validation':val})
    model,epoch,history=fit_graph([quantity_graph(r) for r in train],([quantity_graph(r) for r in val],np.asarray([r['yield_percent'] for r in val])),config,fixed,42,epochs=2)
    save_model(model,config,fixed,output/'model.pt');loaded=load_model(output/'model.pt');expected=predict_record(model,val[0]);actual=predict_record(loaded,val[0])
    if abs(actual-expected)>1e-7:raise AssertionError('Actual smoke checkpoint differs')
    report={'scope':'mechanics-only; not a benchmark or final selection','real_training_rows':len(train),'real_validation_rows':len(val),'epochs':len(history),'history':history,'checkpoint_sha256':sha(output/'model.pt'),'checkpoint_delta':abs(actual-expected),'processed_sha256':sha(data),'test_evaluated':False}
    write(output/'report.json',report);return report


def evaluate(data='data/processed/reactions-quantity-v2.json',output='docs/quantity-v2',plan_path='docs/quantity-plan.json'):
    output=Path(output)
    if (output/'ELN-results.json').exists():raise ValueError('Preserve existing observed ELN evaluation')
    selection=json.loads((output/'selection.json').read_text());plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if sha(data)!=selection['processed_sha256'] or sha(plan_path)!=selection['plan_sha256']:raise ValueError('Frozen data/plan changed')
    for p,digest in selection['feature_sha256'].items():
        if sha('src/reaction_gnn/'+p)!=digest:raise ValueError('Selected representation source changed')
    dataset=json.loads(Path(data).read_text(encoding='utf-8'));dev,test,_=partitions(dataset);config=selection['selected_gnn'];fixed=plan['fixed_training'];version=config['feature_version'];torch.set_num_threads(2)
    graphs=[(old_graph(r) if version==1 else quantity_graph(r)) for r in dev];model,_,history=fit_graph(graphs,None,config,fixed,42,epochs=selection['final_refit_epochs'])
    modeldir=Path('models')/output.name;save_model(model,config,fixed,modeldir/'selected.pt');loaded=load_model(modeldir/'selected.pt')
    test_graphs=[old_graph(r) if version==1 else quantity_graph(r) for r in test];gnn=predictions(loaded,test_graphs,version)
    forest_config=selection['selected_forest'];forest=fit_forest(dev,forest_config);joblib.dump(forest,modeldir/'forest.joblib');forest_prediction=forest_predictions(joblib.load(modeldir/'forest.joblib'),test,forest_config)
    target=np.asarray([r['yield_percent'] for r in test]);median=float(np.median([r['yield_percent'] for r in dev]))
    local=Path('data/interim')/output.name;write(local/'ELN-test-predictions.json',[{'reaction_id':r['reaction_id'],'target':float(y),'gnn':float(p),'forest':float(f)} for r,y,p,f in zip(test,target,gnn,forest_prediction)])
    result={'selection_sha256':sha(output/'selection.json'),'test_status':'Historically observed ELN; retrospective comparison only','test_gnn':metrics(target,gnn),'test_selected_forest':metrics(target,forest_prediction),'test_development_median':metrics(target,np.full(len(target),median)),'gnn_checkpoint_sha256':sha(modeldir/'selected.pt'),'forest_checkpoint_sha256':sha(modeldir/'forest.joblib'),'final_refit_epochs':len(history),'final_refit_history':history,'actual_checkpoint_example_delta':abs(predict_record(model,test[0])-predict_record(loaded,test[0])),'selected_gnn':config,'selected_forest':forest_config}
    write(output/'ELN-results.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['study','smoke','evaluate']);parser.add_argument('--data',default='data/processed/reactions-quantity-v2.json');parser.add_argument('--output');parser.add_argument('--observed-eln',action='store_true');args=parser.parse_args()
    if args.action=='evaluate' and not args.observed_eln:parser.error('evaluate requires --observed-eln acknowledgement')
    output=args.output or ('data/interim/quantity-smoke' if args.action=='smoke' else 'docs/quantity-v2')
    print(json.dumps({'study':study,'smoke':smoke,'evaluate':evaluate}[args.action](args.data,output),indent=2))


if __name__=='__main__':main()
