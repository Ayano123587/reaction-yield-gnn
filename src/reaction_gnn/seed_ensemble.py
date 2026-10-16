"""Prespecified equal-weight seed ensemble selection (development only)."""
import numpy as np


def decide(baseline_fold_mae, ensemble_fold_mae):
    baseline=np.asarray(baseline_fold_mae,dtype=float)
    candidate=np.asarray(ensemble_fold_mae,dtype=float)
    if baseline.shape!=(3,) or candidate.shape!=(3,) or not np.isfinite(baseline).all() or not np.isfinite(candidate).all() or (baseline<0).any() or (candidate<0).any():
        raise ValueError('Three finite nonnegative fold errors required')
    gain=float(baseline.mean()-candidate.mean())
    wins=int((candidate<baseline).sum())
    return {'baseline_mean_mae':float(baseline.mean()),'ensemble_mean_mae':float(candidate.mean()),
            'gain_percentage_points':gain,'fold_wins':wins,
            'adopt_ensemble':bool(gain>=0.5 and wins>=2)}
