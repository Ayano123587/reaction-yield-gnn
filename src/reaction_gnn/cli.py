import argparse
import json
from pathlib import Path
from .data import fetch,prepare

def main(argv=None):
    parser=argparse.ArgumentParser(description="Input-only ORD graph yield regression")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("fetch");p.add_argument("--output",default="data/external/az.parquet");p.add_argument("--provenance",default="docs/dataset-provenance.json")
    p=sub.add_parser("prepare");p.add_argument("--input",default="data/external/az.parquet");p.add_argument("--output",default="data/processed/reactions.json")
    p=sub.add_parser("train");p.add_argument("--input",default="data/processed/reactions.json");p.add_argument("--checkpoint",default="models/yield.pt");p.add_argument("--report",default="docs/results.json");p.add_argument("--epochs",type=int,default=30)
    p=sub.add_parser("optimize");p.add_argument("--input",default="data/processed/reactions.json");p.add_argument("--report",default="docs/optimized-results.json");p.add_argument("--checkpoint",default="models/optimized.pt");p.add_argument("--epochs",type=int,default=60)
    p=sub.add_parser("fetch-transfer");p.add_argument("--output",default="data/external/cn-50688.parquet")
    p=sub.add_parser("transfer-eval");p.add_argument("--input",default="data/external/cn-50688.parquet");p.add_argument("--checkpoint",default="models/optimized.pt");p.add_argument("--development",default="data/processed/reactions.json");p.add_argument("--report",default="docs/source-transfer-results.json");p.add_argument("--limit",type=int,default=2048);p.add_argument("--selection-report",default="docs/optimized-results.json")
    p=sub.add_parser("predict");p.add_argument("checkpoint");p.add_argument("reaction");p.add_argument("--output")
    args=parser.parse_args(argv)
    if args.command=="fetch":
        provenance=fetch(args.output);path=Path(args.provenance);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    elif args.command=="prepare":
        result=prepare(args.input,args.output);print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
    elif args.command=="train":
        from .train import train
        train(args.input,args.checkpoint,args.report,args.epochs)
    elif args.command=="optimize":
        from .optimize import optimize
        optimize(args.input,args.report,args.checkpoint,args.epochs)
    elif args.command=="fetch-transfer":
        from .transfer import fetch_transfer
        print(json.dumps(fetch_transfer(args.output),indent=2))
    elif args.command=="transfer-eval":
        from .transfer import evaluate_transfer
        evaluate_transfer(args.input,args.checkpoint,args.development,args.report,args.limit,args.selection_report)
    else:
        from .train import load_model,predict_record
        value=predict_record(load_model(args.checkpoint),json.loads(Path(args.reaction).read_text(encoding="utf-8")))
        text=json.dumps({"predicted_yield_percent":value,"interpretation":"experimental regression estimate, not a synthesis guarantee"},indent=2)
        if args.output: Path(args.output).write_text(text+"\n",encoding="utf-8")
        print(text)
if __name__=="__main__":main()
