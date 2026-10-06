# Pre-Registration Specification: Picard Iteration vs. Neural SDE Benchmark

**Status:** FROZEN  
**Target Checkpoint:** October 6 Sprint Deliverable  
**Core Rule:** All comparative outcomes reported in the sprint memo are generated exclusively under these pinned configurations. Any hyperparameter adjustments or architectural sweeps performed post-inspection must be categorized under "Exploratory / Ablation Analysis".

---

## 1. Block A: Ordinary Black–Scholes Model (BSM)

### Mathematical Formulation
The European Call option price $V(S, t)$ satisfies the standard Black–Scholes PDE:
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0, \quad V(S, T) = \max(S - K, 0)$$
Or in log-price space $x = \ln(S/K)$, $\tau = T - t$, $u(x, \tau) = e^{r\tau} V/K$:
$$\frac{\partial u}{\partial \tau} = \frac{1}{2}\sigma^2 \frac{\partial^2 u}{\partial x^2} + \left(r - \frac{1}{2}\sigma^2\right)\frac{\partial u}{\partial x}$$

### Pinned Parameters & Rules
1. **Parameter Domain & Sampling:**
   - In-Distribution Hypercube:
     - $S_0 \in [80, 120]$, $K = 100$ (normalized)
     - Maturity $T \in [0.1, 2.0]$ years
     - Volatility $\sigma \in [0.10, 0.50]$
     - Risk-free rate $r \in [0.01, 0.08]$
   - Sampling: Latin Hypercube Sampling (LHS) across the 4D parameter space.
   - Dataset Size: $N_{\text{train}} = 10{,}000$, $N_{\text{val}} = 2{,}000$, $N_{\text{test}} = 2{,}000$ (fixed seeds).
2. **Exact Numerical Reference:**
   - Analytical closed-form Black–Scholes pricing formula evaluated via `scipy.special.ndtr` with double precision (`float64`).
   - Reference tolerance: Machine precision ($\epsilon_{\text{tol}} \approx 10^{-14}$).
3. **Picard Iterative Scheme & Stopping Rule:**
   - Fixed-point integral operator formulation:
     $$u^{(k+1)}(\tau, x) = u^{(0)}(x) + \int_0^\tau \mathcal{L}_{\text{BSM}}[u^{(k)}](s, x) \, ds$$
   - Spatial discretization: Chebyshev collocation grid with $N_x = 200$ nodes on $x \in [-3, 3]$.
   - Time integration: 8th-order Gauss–Legendre quadrature ($N_\tau = 100$ steps).
   - Stopping rule: $\Vert{}u^{(k+1)} - u^{(k)}\Vert{}_\infty < 10^{-6}$ or maximum iterations $K_{\max} = 25$.
4. **Learned Neural SDE (NSDE) Architecture:**
   - Drift network $f_\theta(t, X_t)$: MLP with 3 hidden layers, 64 units/layer, SiLU activations.
   - Diffusion network $g_\phi(t, X_t)$: MLP with 3 hidden layers, 64 units/layer, Softplus output activation (ensuring strictly positive diffusion).
   - SDE Solver: Euler–Maruyama discretization with fixed $\Delta t = T / 100$, 4,096 Monte Carlo sample paths per valuation.
5. **Matched Training & Compute Budget:**
   - Optimizer: AdamW ($\text{lr} = 10^{-3}$, cosine decay to $10^{-5}$, weight decay $10^{-4}$).
   - Batch size: 256.
   - Hard training budget: 100 epochs (approx. 3,900 gradient steps) or a hard compute cap of 15 minutes on 1x NVIDIA GPU / matched CPU.
6. **Random Seeds:**
   - Master seed: `42` (pinned across Python `random`, NumPy, PyTorch, and CUDA deterministic flags).
7. **Pricing Error & Convergence Metrics:**
   - Pricing Error: Root Mean Squared Error (RMSE), Mean Absolute Relative Error (MARE: $\frac{1}{N}\sum \frac{\vert{}\hat{V} - V^*\vert{}}{V^*}$), and Max Absolute Error ($L_\infty$).
   - Convergence: Picard contraction quotient $\rho_k = \frac{\Vert{}u^{(k+1)} - u^{(k)}\Vert{}_\infty}{\Vert{}u^{(k)} - u^{(k-1)}\Vert{}_\infty}$; NSDE epoch-wise loss and validation RMSE curves.
8. **Runtime Measurement:**
   - 10 warm-up evaluations followed by 50 timed runs using `time.perf_counter_ns()`. GPU synchronized via `torch.cuda.synchronize()`. Report median and Interquartile Range (IQR).
9. **Extrapolation Regime (Out-of-Distribution):**
   - Moneyness: Deep OTM/ITM $S_0 \in [40, 79] \cup [121, 180]$.
   - Maturity: $T \in [2.5, 5.0]$ years.
   - Extreme Volatility: $\sigma \in [0.55, 1.20]$.
10. **Explicit Failure Criteria:**
    - Divergence: Output containing NaN, Inf, or $\vert{}V\vert{} > 10^4$.
    - Arbitrage violation: Negative option price ($V < 0$), lower bound violation ($V < \max(0, S - K e^{-rT})$), or negative delta ($\partial V/\partial S < 0$).
    - Stalling: Picard iteration failing to reach $10^{-6}$ tolerance within 25 steps; NSDE validation relative error $> 10\%$ at epoch 100.

---

## 2. Block B: Time-Fractional Black–Scholes Model (Fractional-Grid)

### Mathematical Formulation
The subdiffusive time-fractional Black–Scholes PDE with Caputo fractional derivative of order $\alpha \in (0, 1]$:
$${}^C D_t^\alpha V + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0, \quad V(S, T) = \max(S - K, 0)$$
where ${}^C D_t^\alpha V(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t (t - s)^{-\alpha} \frac{\partial V}{\partial s} \, ds$.

### Pinned Parameters & Rules
1. **Parameter Domain & Sampling:**
   - In-Distribution Hypercube:
     - $S_0 \in [80, 120]$, $K = 100$
     - Maturity $T \in [0.1, 1.5]$ years
     - Volatility $\sigma \in [0.15, 0.45]$
     - Risk-free rate $r \in [0.02, 0.06]$
     - Fractional order $\alpha \in [0.60, 0.95]$
   - Sampling: Latin Hypercube Sampling (LHS) across the 5D parameter space.
   - Dataset Size: $N_{\text{train}} = 10{,}000$, $N_{\text{val}} = 2{,}000$, $N_{\text{test}} = 2{,}000$.
2. **Exact Numerical Reference:**
   - High-resolution finite difference solver utilizing L1 discretization for the Caputo derivative on a graded temporal mesh ($N_t = 1{,}000$, grading parameter $r_{\text{mesh}} = 2.0$) and second-order central spatial differences ($N_s = 1{,}000$).
   - Reference verification tolerance: Relative Richardson extrapolation error $\Vert{}V_{\Delta t} - V_{\Delta t/2}\Vert{}_\infty / \Vert{}V_{\Delta t}\Vert{}_\infty < 10^{-5}$.
3. **Fractional Picard Iterative Scheme & Stopping Rule:**
   - Volterra fractional integral formulation:
     $$u^{(k+1)}(t, x) = u(0, x) + \frac{1}{\Gamma(\alpha)} \int_0^t (t - s)^{\alpha - 1} \mathcal{L}[u^{(k)}](s, x) \, ds$$
   - Discrete evaluation: Product integration / fractional Adams–Bashforth–Moulton scheme on $N_t = 100$ temporal grid points.
   - Stopping rule: $\Vert{}u^{(k+1)} - u^{(k)}\Vert{}_\infty < 10^{-5}$ or maximum iterations $K_{\max} = 30$.
4. **Learned Fractional NSDE Architecture:**
   - Continuous-time Markovian approximation / Volterra-type Neural SDE conditioned on $[t, X_t, \alpha, \sigma, r]$.
   - Network: 4 hidden layers, 64 units/layer, SiLU activations.
   - Numerical integration: Euler–Maruyama scheme with Cholesky decomposition of the fractional kernel covariance, 4,096 Monte Carlo paths.
5. **Matched Training & Compute Budget:**
   - Optimizer: AdamW ($\text{lr} = 10^{-3}$, cosine decay, weight decay $10^{-4}$).
   - 100 epochs, batch size 256 (identical step count and GPU wall-clock cap as Block A).
6. **Random Seeds:**
   - Master seed: `42`.
7. **Pricing Error & Convergence Metrics:**
   - RMSE, MARE, $L_\infty$ error against the high-resolution L1 numerical reference.
   - Picard empirical contraction rate: $\hat{L}(\alpha) = \frac{\Vert{}u^{(k+1)} - u^{(k)}\Vert{}_\infty}{\Vert{}u^{(k)} - u^{(k-1)}\Vert{}_\infty}$ recorded across fractional orders $\alpha$.
8. **Runtime Measurement:**
   - 10 warm-up runs, 50 timed evaluations via `time.perf_counter_ns()`, GPU synchronized. Median and IQR reported.
9. **Extrapolation Regime (Out-of-Distribution):**
   - Fractional Order Extremes: $\alpha \in [0.30, 0.55] \cup [0.96, 0.99]$.
   - Moneyness: $S_0 \in [50, 79] \cup [121, 160]$.
   - Maturity: $T \in [1.6, 3.0]$ years.
10. **Explicit Failure Criteria:**
    - Kernel singularity blow-up ($s \to t$) or NaN/Inf values.
    - Loss of contractivity in Picard iteration ($\hat{L} \ge 1.0$).
    - Monotonicity violation with respect to fractional order ($\partial V/\partial \alpha$ non-smooth or unphysical oscillation).
