import numpy as np
from scipy.special import gamma
from scipy.linalg import solve_banded


def solve_tfbsm_l1(S0, K, T, r, sigma, alpha, Nt=2000, Ns=2000, S_max_mult=4.0):
    """
    High-precision L1 Finite Difference solver on graded temporal mesh
    for Caputo time-fractional Black-Scholes equation.
    """
    S_max = K * S_max_mult
    S_grid = np.linspace(0.0, S_max, Ns + 1)
    dS = S_grid[1] - S_grid[0]

    # Graded time mesh to resolve initial boundary singularity
    gamma_mesh = (2.0 - alpha) / alpha
    j_idx = np.arange(Nt + 1)
    tau_grid = T * (j_idx / Nt) ** gamma_mesh
    dtau = np.diff(tau_grid)
    c_alpha = 1.0 / gamma(2.0 - alpha)

    # Initial condition: payoff at tau = 0
    V = np.maximum(S_grid - K, 0.0)
    history_V = [V.copy()]

    i_idx = np.arange(1, Ns)
    S_int = S_grid[i_idx]

    # Spatial differential operator coefficients
    a_i = 0.5 * (sigma**2 * (S_int**2) / (dS**2) - r * S_int / dS)
    b_i = -(sigma**2 * (S_int**2) / (dS**2) + r)
    c_i = 0.5 * (sigma**2 * (S_int**2) / (dS**2) + r * S_int / dS)

    for n in range(1, Nt + 1):
        dt_n = dtau[n - 1]
        a0 = (dt_n ** (-alpha)) * c_alpha

        diagonals = np.zeros((3, Ns - 1))
        diagonals[0, 1:] = -c_i[:-1]   # Upper diagonal
        diagonals[1, :] = a0 - b_i     # Main diagonal
        diagonals[2, :-1] = -a_i[1:]   # Lower diagonal

        # Assemble Caputo memory history
        rhs = np.zeros(Ns - 1)
        for k in range(n):
            ak_k = ((tau_grid[n] - tau_grid[k])**(1.0 - alpha) - 
                    (tau_grid[n] - tau_grid[k+1])**(1.0 - alpha)) / dtau[k] * c_alpha
            rhs += ak_k * (history_V[k+1][1:Ns] - history_V[k][1:Ns])

        rhs = a0 * history_V[n-1][1:Ns] - rhs

        # Boundary conditions
        V_0 = 0.0
        V_max = S_max - K * np.exp(-r * tau_grid[n])
        rhs[0] += a_i[0] * V_0
        rhs[-1] += c_i[-1] * V_max

        V_interior = solve_banded((1, 1), diagonals, rhs)
        V_new = np.zeros(Ns + 1)
        V_new[0] = V_0
        V_new[1:Ns] = V_interior
        V_new[-1] = V_max
        history_V.append(V_new)

    return float(np.interp(S0, S_grid, history_V[-1]))
