.PHONY: install data train test fetch-transfer
install:
	python -m pip install -e ".[dev,notebooks]"
data:
	reaction-gnn fetch
	reaction-gnn prepare
	python -c "from reaction_gnn.quantities import prepare_quantitative; prepare_quantitative('data/external/az.parquet','data/processed/reactions-quantity-v2.json')"

# Change RUN_DIR for each new validation study; completed runs are never replaced.
RUN_DIR ?= docs/manual-run
train:
	python train.py --output "$(RUN_DIR)"
test:
	python -m pytest -q
fetch-transfer:
	reaction-gnn fetch-transfer
