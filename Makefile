.PHONY: install data train test fetch-transfer
install:
	python -m pip install -e ".[dev,notebooks]"
data:
	reaction-gnn fetch
	reaction-gnn prepare
	python -c "from reaction_gnn.quantities import prepare_quantitative; prepare_quantitative('data/external/az.parquet','data/processed/reactions-quantity-v2.json')"

# Use a fresh directory for each selected-model reproduction.
RUN_DIR ?= models/manual-run
train:
	python train.py --output "$(RUN_DIR)"
test:
	python -m pytest -q
fetch-transfer:
	reaction-gnn fetch-transfer
