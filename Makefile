install:
	python -m pip install -e ".[dev]"
data:
	reaction-gnn fetch
	reaction-gnn prepare
train:
	reaction-gnn train --epochs 30
test:
	python -m pytest -q
optimize:
	reaction-gnn optimize --epochs 60
transfer:
	reaction-gnn fetch-transfer
	reaction-gnn transfer-eval --limit 2048
