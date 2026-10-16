# Model artifacts

Training saves a weights-only-compatible PyTorch checkpoint with architecture width and graph feature version. Run `python train.py --output models/my-model` from the repository root to create `models/my-model/selected.pt`; the output directory must be fresh. Binary weights are ignored; aggregate histories and metrics are committed under docs/.
