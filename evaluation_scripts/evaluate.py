"""Refit a frozen selection and explicitly score the historically observed ELN holdout."""
import argparse
import json
from reaction_gnn.quantity_study import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/reactions-quantity-v2.json')
    parser.add_argument('--selection-directory', required=True)
    parser.add_argument('--acknowledge-observed-eln', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.selection_directory), indent=2))
