import numpy as np
from scipy.stats import kendalltau

def calculate_kendall_tau(predicted_scores, ground_truth_ranks, lengths):
    taus = []
    for pred, gt, n in zip(predicted_scores, ground_truth_ranks, lengths):
        n = int(n)
        if n < 2: #skip tiny videos
            continue
        tau, _ = kendalltau(pred[:n], gt[:n]) #compare ground truth vs predicted score
        if np.isnan(tau):
            tau = 0.0
        taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0
