"""Run development-only experiments; never score the ELN holdout."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['quantity', 'morgan', 'loss'], default='quantity')
    parser.add_argument('--data', default='data/processed/reactions-quantity-v2.json')
    parser.add_argument('--source', help='Frozen preceding selection directory for followup studies')
    parser.add_argument('--output', required=True, help='New empty aggregate report directory')
    args = parser.parse_args()
    if args.study == 'quantity':
        if args.source:
            parser.error('--source applies only to followup studies')
        from reaction_gnn.quantity_study import study
        study(args.data, args.output)
    else:
        if args.data != 'data/processed/reactions-quantity-v2.json':
            parser.error('Followups use the frozen documented processed dataset')
        if args.study == 'morgan':
            from reaction_gnn.morgan import run
            source = args.source or 'docs/quantity-v2'
        else:
            from reaction_gnn.loss_study import run
            source = args.source or 'docs/morgan-v3'
        run(source, args.output)
