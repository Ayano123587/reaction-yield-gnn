"""Run validation-only quantity experiments; never score the ELN holdout."""
import argparse
from reaction_gnn.quantity_study import study

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/reactions-quantity-v2.json')
    parser.add_argument('--output', required=True, help='New empty aggregate report directory')
    args = parser.parse_args()
    study(args.data, args.output)
