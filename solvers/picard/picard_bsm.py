import numpy as np
from scipy.ndimage import convolve1d


class PicardBSMSolver:
    """Integral-operator Picard iteration using the Gaussian heat kernel."""
    def __init__(self, Nx=1000, Ntau=200, x_lim=(-2.5, 2.5), tol=1e-7, max_iter=50):
        self.Nx = Nx
        self.Ntau = Ntau
        self.x = np.linspace(x_lim[0], x_lim[1], Nx)
        self.dx = self.x[1] - self.x[0]
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, S, K, T, r, sigma):
        tau = np.linspace(0.0, T, self.Ntau)
        dtau = tau[1] - tau[0] if self.Ntau > 1 else T

        u0 = np.maximum(np.exp(self.x) - 1.0, 0.0)
        u_curr = np.tile(u0, (self.Ntau, 1))
        drift_coeff = r - 0.5 * sigma**2

        diff_history = []
        for _ in range(self.max_iter):
            u_next = np.zeros_like(u_curr)
            u_next[0] = u0

            du_dx = np.gradient(u_curr, self.dx, axis=1)

            for n in range(1, self.Ntau):
                t_diff = np.maximum(tau[n] - tau[:n], 1e-10)
                var = np.maximum(sigma**2 * t_diff[:, None], 1e-12)
                kernel = np.exp(-(self.x[None, :]**2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)
                kernel /= np.sum(kernel, axis=1, keepdims=True) * self.dx

                drift_term = drift_coeff * du_dx[:n]
                conv = np.array([convolve1d(drift_term[m], kernel[m], mode="nearest") for m in range(n)])
                u_next[n] = u0 + dtau * np.sum(conv, axis=0)

            diff = np.max(np.abs(u_next - u_curr))
            diff_history.append(diff)
            if diff < self.tol:
                u_curr = u_next
                break
            u_curr = u_next

        x_target = np.log(S / K)
        u_val = np.interp(x_target, self.x, u_curr[-1])
        price = float(K * np.exp(-r * T) * u_val)
        contraction = float(diff_history[-1] / diff_history[-2]) if len(diff_history) > 1 else 0.0
        return price, len(diff_history), contraction
