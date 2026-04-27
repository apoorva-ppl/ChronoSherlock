"""
Kendall-tau validation metric.

kendalltau is rank-invariant: it doesn't matter whether we feed it
float ranks or integer positional orders, as long as both sides are
consistent. We just pass predicted scores vs. ground-truth ranks
directly — scipy handles the rest.
"""
import numpy as np
from scipy.stats import kendalltau


def calculate_kendall_tau(predicted_scores, ground_truth_ranks, lengths):
    """
    predicted_scores   : list[np.ndarray]  — raw model scores per video
    ground_truth_ranks : list[np.ndarray]  — normalised [0,1] target ranks
    lengths            : list[int]         — valid-frame count per video
    """
    taus = []
    for pred, gt, n in zip(predicted_scores, ground_truth_ranks, lengths):
        n = int(n)
        if n < 2:
            continue
        tau, _ = kendalltau(pred[:n], gt[:n])
        if np.isnan(tau):
            tau = 0.0
        taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0
