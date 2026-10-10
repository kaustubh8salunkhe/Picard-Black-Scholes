import numpy as np
from scipy.special import gamma


class PicardTFBSMSolver:
    """Product-quadrature Volterra-Picard iteration for Caputo fractional BSM."""
    def __init__(self, Nt=100, Ns=500, tol=1e-6, max_iter=40):
        self.Nt = Nt
        self.Ns = Ns
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, S0, K, T, r, sigma, alpha):
        S_max = 3.0 * K
        S = np.linspace(0.0, S_max, self.Ns + 1)
        dS = S[1] - S[0]
        tau = np.linspace(0.0, T, self.Nt + 1)

        V0 = np.maximum(S - K, 0.0)
        V_curr = np.tile(V0, (self.Nt + 1, 1))

        def apply_lbs(V_mat):
            d2V = (V_mat[:, 2:] - 2.0 * V_mat[:, 1:-1] + V_mat[:, :-2]) / (dS**2)
            dV = (V_mat[:, 2:] - V_mat[:, :-2]) / (2.0 * dS)
            S_int = S[1:-1]
            LV = 0.5 * (sigma**2) * (S_int**2) * d2V + r * S_int * dV - r * V_mat[:, 1:-1]
            out = np.zeros_like(V_mat)
            out[:, 1:-1] = LV
            return out

        diff_history = []
        for _ in range(self.max_iter):
            LV = apply_lbs(V_curr)
            V_next = np.zeros_like(V_curr)
            V_next[0] = V0

            for n in range(1, self.Nt + 1):
                j = np.arange(n)
                weights = ((tau[n] - tau[j])**alpha - (tau[n] - tau[j+1])**alpha) / alpha
                integral_sum = np.sum(weights[:, None] * LV[:n], axis=0) / gamma(alpha)
                V_next[n] = V0 + integral_sum

            diff = np.max(np.abs(V_next - V_curr))
            diff_history.append(diff)
            if diff < self.tol:
                V_curr = V_next
                break
            V_curr = V_next

        price = float(np.interp(S0, S, V_curr[-1]))
        contraction = float(diff_history[-1] / diff_history[-2]) if len(diff_history) > 1 else 0.0
        return price, len(diff_history), contraction
