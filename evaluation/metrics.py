import numpy as np


def compute_rmse(pred, ref):
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def compute_mape(pred, ref, threshold=0.50):
    mask = ref >= threshold
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((pred[mask] - ref[mask]) / ref[mask])) * 100.0)


def compute_max_ae(pred, ref):
    return float(np.max(np.abs(pred - ref)))


def audit_failures(prices, S, K, T, r):
    prices = np.asarray(prices, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    is_nan_inf = np.isnan(prices) | np.isinf(prices)
    lower_bound = np.maximum(S - K * np.exp(-r * T), 0.0) - 1e-4
    upper_bound = S + 1e-4

    arb_violation = (prices < lower_bound) | (prices > upper_bound)
    return {
        "nan_inf_rate": float(np.mean(is_nan_inf)),
        "arbitrage_violation_rate": float(np.mean(arb_violation)),
        "is_nan_inf": is_nan_inf,
        "arbitrage_violation": arb_violation,
    }
