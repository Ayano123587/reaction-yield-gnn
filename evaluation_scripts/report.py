"""Read aggregate results without training, loading data, or evaluating a holdout."""
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    for name in ('gnn-validation.json', 'selection.json', 'ELN-results.json'):
        path = args.directory / name
        if path.exists():
            print(name)
            print(json.dumps(json.loads(path.read_text(encoding='utf-8')), indent=2))
