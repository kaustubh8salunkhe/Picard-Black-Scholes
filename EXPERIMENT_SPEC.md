
# EXPERIMENT_SPEC.md: Frozen Benchmark Specification

**Status:** FROZEN

**Commit Target:** Pre-inspection Baseline

**Scope:** Picard Iteration vs. Neural SDE Benchmarking (Ordinary BSM and Time-Fractional BSM)

---

## 1. Governance and Protocol Rules

1. **Pre-Registration Freeze:** All data sampling, network architectures, solver tolerances, training budgets, and evaluation routines defined in this specification are locked prior to inspecting comparative outcomes.
2. **Exploratory Policy:** Any hyperparameter tuning, loss function adjustment, or architectural modification executed after inspecting initial baseline outcomes must not overwrite these baselines. Such additions must be committed under branch/tag `exploratory/*` and documented in a separate section of the October 6 research memo.
3. **Model Separation:** Block A (Ordinary Black–Scholes–Merton) and Block B (Time-Fractional Black–Scholes) are treated as distinct experimental blocks. Solvers, references, and surrogate models are isolated into separate execution pipelines.

---

## 2. Block A: Ordinary Black–Scholes–Merton (BSM)

### 2.1 Governing Equation and Boundary Conditions

Standard European call option value $V(t, S)$ governed by:


$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

Boundary conditions:

* $V(T, S) = \max(S - K, 0)$
* $V(t, 0) = 0$
* $\lim_{S \to \infty} [V(t, S) - (S - K e^{-r(T-t)})] = 0$

### 2.2 Numerical Reference (Ground Truth)

* **Solver:** Analytical Black–Scholes formula in double precision (`float64`).
* **Cumulative Normal Implementation:** High-precision polynomial approximation (`scipy.special.ndtr` / standard Faddeeva-based implementation).
* **Reference Tolerance:** Machine epsilon ($< 10^{-15}$).

### 2.3 Picard Iteration Formulation

* **Transformation:** Log-moneyness $x = \ln(S/K)$, time-to-maturity $\tau = T - t$, $u(\tau, x) = e^{r\tau} V/K$.
* **Integral Operator:** Inverted heat operator using the Gaussian kernel $G(\tau, x)$:

$$u^{(k+1)}(\tau, x) = u^{(0)}(x) + \int_0^\tau \int_{-\infty}^{\infty} G(\tau - s, x - y) \mathcal{N}\left(u^{(k)}(s, y)\right) dy \, ds$$


* **Spatial Discretization:** Uniform grid on $x \in [-2.5, 2.5]$ with $N_x = 1000$ points.
* **Temporal Integration:** Trapezoidal quadrature with $N_\tau = 200$ time steps.
* **Stopping Criterion:**

$$\Vert{}u^{(k+1)} - u^{(k)}\Vert{}_\infty < 10^{-7} \quad \text{or} \quad k \ge 50$$



### 2.4 Learned Neural SDE Architecture

* **Formulation:** Latent SDE with risk-neutral drift dynamics and neural diffusion:

$$d X_t = r X_t dt + \sigma_\theta(t, X_t) X_t dW_t, \quad V_\theta(0, S_0) = e^{-rT} \mathbb{E}\left[\max(X_T - K, 0)\right]$$


* **Discretization:** Euler–Maruyama scheme with $N_t = 100$ uniform intervals.
* **Network Backbone ($\sigma_\theta$):**
* Topology: Multi-Layer Perceptron (MLP)
* Layers: 3 hidden layers, 64 hidden units per layer
* Activation: SiLU (Swish)
* Output Layer: Linear with Softplus activation $+ 10^{-4}$ (strictly positive diffusion)


* **Training Objective:** Mean Squared Error on analytical BSM prices across training samples:

$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( \hat{V}_\theta(t_i, S_i) - V_{\text{analytical}}(t_i, S_i) \right)^2$$



### 2.5 Parameter Domain and Sampling

Fixed Strike: $K = 100$.

| Parameter | Interpolation Domain (Train / Val / Test) | Extrapolation Domain (Stress Testing) |
| --- | --- | --- |
| Spot Price ($S$) | $[60, 140]$ | $[30, 60) \cup (140, 200]$ |
| Volatility ($\sigma$) | $[0.10, 0.40]$ | $[0.45, 0.80]$ |
| Maturity ($T$) | $[0.1, 2.0]$ years | $[2.1, 4.0]$ years |
| Risk-free Rate ($r$) | $[0.01, 0.08]$ | $[0.08, 0.15]$ |

* **Sampling Scheme:** Sobol quasi-random sequence.
* **Dataset Sizes:**
* Training: 10,000 points
* Validation: 2,000 points
* In-Domain Test: 5,000 points
* Out-of-Distribution (OOD) Extrapolation Test: 2,500 points



---

## 3. Block B: Time-Fractional Black–Scholes Model (TFBSM)

### 3.1 Governing Equation

Time-fractional Black–Scholes equation featuring a Caputo fractional time derivative of order $\alpha \in (0, 1]$ and standard spatial diffusion:


$${}_0^C D_t^\alpha V(t, S) + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

Where the Caputo derivative is defined as:


$${}_0^C D_t^\alpha V(t, S) = \frac{1}{\Gamma(1-\alpha)} \int_0^t (t - s)^{-\alpha} \frac{\partial V(s, S)}{\partial s} ds$$

### 3.2 Numerical Reference (Ground Truth)

* **Temporal Scheme:** L1 approximation of the Caputo fractional derivative on a graded mesh:

$$t_j = T \left(\frac{j}{N_t}\right)^r, \quad r = (2-\alpha)/\alpha, \quad N_t = 2000$$


* **Spatial Scheme:** Second-order central finite differences on non-uniform log-space grid ($N_S = 2000$).
* **Linear System Solver:** Tridiagonal matrix algorithm (Thomas algorithm) at each time level.
* **Convergence Verification:** Mesh refinement study verifying temporal order $\mathcal{O}(\Delta t^{2-\alpha})$ and spatial order $\mathcal{O}(\Delta x^2)$.
* **Reference Tolerance:** Discretization tolerance set such that grid-doubling relative difference satisfies:

$$\frac{\Vert{}V_{2N} - V_N\Vert{}_2}{\Vert{}V_N\Vert{}_2} < 5 \times 10^{-7}$$



### 3.3 Picard Iteration Formulation

* **Fractional Volterra Representation:**

$$V(t, S) = V(0, S) + \frac{1}{\Gamma(\alpha)} \int_0^t (t - s)^{\alpha - 1} \mathcal{L}_{BS} V(s, S) \, ds$$



where $\mathcal{L}_{BS} = -\left(\frac{1}{2}\sigma^2 S^2 \partial_{SS} + r S \partial_S - r I\right)$.
* **Iterative Step:**

$$V^{(k+1)}(t_n, \cdot) = V(0, \cdot) + \frac{1}{\Gamma(\alpha)} \sum_{j=0}^{n-1} w_{j, n}^{(\alpha)} \mathcal{L}_{BS} V^{(k)}(t_j, \cdot)$$



using product trapezoidal quadrature weights $w_{j, n}^{(\alpha)}$.
* **Stopping Criterion:**

$$\max_{n} \Vert{}V^{(k+1)}(t_n, \cdot) - V^{(k)}(t_n, \cdot)\Vert{}_\infty < 10^{-6} \quad \text{or} \quad k \ge 40$$



### 3.4 Learned Neural Volterra / Fractional NSDE Architecture

* **Formulation:** Discretized Volterra SDE incorporating a power-law convolutional kernel:

$$X_t = X_0 + \int_0^t b_\theta(s, X_s) ds + \int_0^t \frac{(t-s)^{\alpha - 1/2}}{\Gamma(\alpha + 1/2)} \sigma_\phi(s, X_s) dW_s$$


* **Discretization:** Hybrid convolution-Euler scheme evaluated over $N_t = 128$ steps.
* **Architecture:**
* Drift network $b_\theta$: 3 hidden layers, 64 units, SiLU
* Volatility network $\sigma_\phi$: 3 hidden layers, 64 units, SiLU, Softplus output activation


* **Kernel Order ($\alpha$):** Explicit scalar input appended to coordinates $(t, X_t, \alpha)$.

### 3.5 Parameter Domain and Sampling

Fixed Strike: $K = 100$.

| Parameter | Interpolation Domain (Train / Val / Test) | Extrapolation Domain (Stress Testing) |
| --- | --- | --- |
| Fractional Order ($\alpha$) | $[0.45, 0.95]$ | $[0.15, 0.40)$ (Extreme memory regime) |
| Spot Price ($S$) | $[70, 130]$ | $[40, 70) \cup (130, 180]$ |
| Volatility ($\sigma$) | $[0.15, 0.40]$ | $[0.45, 0.70]$ |
| Maturity ($T$) | $[0.2, 1.5]$ years | $[1.6, 3.0]$ years |
| Risk-free Rate ($r$) | $[0.02, 0.06]$ | $[0.06, 0.10]$ |

* **Sampling Scheme:** Sobol sequence across 5-dimensional space $(S, \sigma, T, r, \alpha)$.
* **Dataset Sizes:**
* Training: 15,000 points
* Validation: 3,000 points
* In-Domain Test: 5,000 points
* Extrapolation Test: 3,000 points



---

## 4. Shared Training and Compute Budget

* **Hardware Target:** Single NVIDIA RTX 4090 / A100 GPU (or pinned 8-core CPU allocation for CPU reference benchmarks).
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
* **Learning Rate Schedule:** Initial $\eta_0 = 10^{-3}$, cosine decay down to $\eta_{\min} = 10^{-5}$ without restarts.
* **Batch Size:** 256.
* **Total Epochs:** 1,000 epochs (fixed; no early stopping to preserve matched compute trajectories).
* **RNG Seeds:**
* Dataset Partitioning & Sampling: `Seed = 42`
* Model Weight Initialization: `Seed = 1337`
* Evaluation Path Generation: `Seed = 2026`



---

## 5. Pricing Error and Convergence Metrics

1. **Root Mean Squared Error (RMSE):**

$$\text{RMSE} = \sqrt{\frac{1}{M} \sum_{m=1}^M \left( \hat{V}_m - V_m^{\text{ref}} \right)^2}$$


2. **Mean Absolute Percentage Error (MAPE):**

$$\text{MAPE} = \frac{100\%}{M} \sum_{m=1}^M \left\vert{} \frac{\hat{V}_m - V_m^{\text{ref}}}{V_m^{\text{ref}}} \right\vert{} \quad (\text{filtered for } V_m^{\text{ref}} \ge 0.50)$$


3. **Maximum Absolute Error (MaxAE):**

$$\text{MaxAE} = \max_{1 \le m \le M} \vert{}\hat{V}_m - V_m^{\text{ref}}\vert{}$$


4. **Picard Contraction Rate Metric:**

$$\hat{L}_k = \frac{\Vert{}V^{(k+1)} - V^{(k)}\Vert{}_\infty}{\Vert{}V^{(k)} - V^{(k-1)}\Vert{}_\infty}$$



---

## 6. Runtime Measurement Protocol

1. **Execution Mode:** Isolated process pinned to dedicated threads/device.
2. **Warm-Up:** 20 complete evaluation passes discarded prior to logging.
3. **Measurement:** 100 consecutive runs timed using `time.perf_counter_ns()`.
4. **GPU Synchronization:** `torch.cuda.synchronize()` explicitly invoked before and after timer boundaries.
5. **Reported Statistics:** Median execution time, 5th percentile, and 95th percentile latency (milliseconds per 1,000 pricing evaluations).

---

## 7. Explicit Failure Criteria

A solver run or model output is flagged as a **Failure** if it triggers any of the following:

1. **Numerical Breakdown:** Generation of `NaN`, `Inf`, or floating-point overflow during training, iteration, or inference.
2. **Arbitrage Bound Violation:**
* Violation of lower call bound: $\hat{V} < \max(S - K e^{-rT}, 0) - 10^{-4}$
* Violation of upper bound: $\hat{V} > S + 10^{-4}$


3. **Monotonicity Violation (Delta Anomaly):**

$$\frac{\partial \hat{V}}{\partial S} < -10^{-3} \quad \text{or} \quad \frac{\partial \hat{V}}{\partial S} > 1 + 10^{-3}$$


4. **Picard Divergence:** Spectral growth $\hat{L}_k \ge 1.0$ for three consecutive iterations, or $\Vert{}V^{(k)}\Vert{}_\infty > 10 \cdot S_0$.
5. **Runtime Timeout:** Execution latency exceeding 10x the median reference solver runtime for a matched test batch.
