import numpy as np

def calculate_ece(y_true, y_proba, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            bin_proba = y_proba[mask]          # Probabilidades no bin
            bin_true = y_true[mask]            # Labels reais
            avg_pred = np.mean(bin_proba)      # conf(B_m)
            avg_true = np.mean(bin_true)       # acc(B_m)
            ece += np.abs(avg_pred - avg_true) * len(bin_proba) / len(y_true)
    
    return ece