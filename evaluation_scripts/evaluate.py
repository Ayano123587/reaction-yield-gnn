"""Explicit observed ELN evaluation of an already trained selected checkpoint."""
import argparse
import json
from reaction_gnn.selected import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/reactions-quantity-v2.json')
    parser.add_argument('--checkpoint-directory', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--acknowledge-observed-eln', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.checkpoint_directory, args.report,
                             acknowledge_observed=args.acknowledge_observed_eln), indent=2))
