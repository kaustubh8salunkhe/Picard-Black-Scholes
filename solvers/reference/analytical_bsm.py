import numpy as np
from scipy.special import ndtr


def bsm_call_price(S, K, T, r, sigma):
    """Closed-form analytical European call pricing in float64 precision."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    sqrt_T = np.sqrt(np.maximum(T, 1e-14))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    return S * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
