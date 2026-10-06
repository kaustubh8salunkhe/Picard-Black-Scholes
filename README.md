# Picard vs. Neural SDE: Option Pricing Benchmark

A reproducible computational benchmark comparing **Picard Iterative Methods** against **Learned Neural SDEs (NSDE)** for European call option pricing under standard Black–Scholes–Merton (BSM) formulation.

This repository implements the pre-registered experimental configuration defined in `EXPERIMENT_SPEC.md` for the October 6 research sprint checkpoint.

---

## Repository Structure

```text
├── EXPERIMENT_SPEC.md        # Frozen experimental protocol, architectures, and metrics
├── README.md                 # Setup, usage, and pipeline reproduction guide
├── configs/
│   ├── block_a_bsm.yaml      # Configuration for Block A (Ordinary BSM)
│   └── block_b_tfbsm.yaml    # Configuration for Block B (Time-Fractional BSM)
├── data/
│   ├── generate_sobol.py     # Sobol quasi-random sampling for in-domain & OOD grids
│   └── datasets/             # Generated train/val/test/extrapolation CSV files
├── solvers/
│   ├── reference/
│   │   ├── analytical_bsm.py # Closed-form Black–Scholes analytical reference
│   │   └── l1_fdm_tfbsm.py   # L1 graded-mesh finite-difference solver (Caputo time-fractional)
│   ├── picard/
│   │   ├── picard_bsm.py     # Integral operator Picard iteration for standard BSM
│   │   └── picard_tfbsm.py   # Volterra-Picard scheme with product trapezoidal weights
│   └── neural_sde/
│       ├── models.py         # MLP drift/diffusion parameterizations
│       ├── engine.py         # Euler–Maruyama and convolutional Volterra integrators
│       └── train.py          # Fixed-budget training loops (AdamW, cosine decay)
├── evaluation/
│   ├── benchmark_runner.py   # Unified execution harness across solvers
│   ├── metrics.py            # RMSE, MAPE, MaxAE, Picard contraction, and failure flags
│   ├── timing.py             # CUDA/CPU latency measurement with synchronization
│   └── visualizer.py         # Convergence curves and error surface generators
├── outputs/
│   ├── raw/                  # Raw inference, pricing, and error CSV dumps
│   └── figures/              # Generated error surfaces, convergence, and OOD plots
└── memo/
    └── research_memo.tex     # 1–2 page LaTeX technical research memo

```

---

## Getting Started

### Prerequisites

* Python 3.10+
* CUDA 12.0+ (optional, fallback to CPU supported)

### Installation

Clone the repository and install dependencies in an isolated virtual environment:

```bash
git clone <repo-url>
cd picard-nsde-benchmark

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

```

#### Core Dependencies (`requirements.txt`)

```text
torch>=2.2.0
scipy>=1.11.0
numpy>=1.26.0
pandas>=2.1.0
pyyaml>=6.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0

```

---

## Experimental Workflow

Reproduce the complete evaluation pipeline end-to-end via the following sequential stages:

### Step 1: Generate Frozen Datasets

Generate Sobol quasi-random parameter sets for both interpolation and extrapolation regimes:

```bash
# Generates data/datasets/block_a_*.csv and data/datasets/block_b_*.csv
python data/generate_sobol.py --config configs/block_a_bsm.yaml
python data/generate_sobol.py --config configs/block_b_tfbsm.yaml

```

### Step 2: Train Neural SDE Surrogates

Run fixed-budget training (1,000 epochs, cosine decay, zero early stopping) under pinned seeds:

```bash
# Block A: Standard NSDE
python solvers/neural_sde/train.py --config configs/block_a_bsm.yaml --seed 1337

# Block B: Time-Fractional Volterra NSDE
python solvers/neural_sde/train.py --config configs/block_b_tfbsm.yaml --seed 1337

```

### Step 3: Execute Benchmark Evaluation

Evaluate Picard iteration schemes against the trained Neural SDE models using ground-truth references:

```bash
# Run Block A benchmark (Analytical BSM reference)
python evaluation/benchmark_runner.py --block A --output-dir outputs/raw/

# Run Block B benchmark (L1 Graded Finite Difference reference)
python evaluation/benchmark_runner.py --block B --output-dir outputs/raw/

```

### Step 4: Generate Diagnostic Artifacts

Generate convergence plots, error surfaces, and failure audits:

```bash
python evaluation/visualizer.py \
  --results-dir outputs/raw/ \
  --output-dir outputs/figures/

```

---

## Parameter Blocks Summary

| Attribute | Block A (Ordinary BSM) | Block B (Time-Fractional BSM) |
| --- | --- | --- |
| **Equation** | Classical Black–Scholes PDE | Caputo Time-Fractional PDE (${}_0^C D_t^\alpha$) |
| **Reference Ground Truth** | Analytical formula (`scipy.special.ndtr`) | L1 scheme on graded mesh ($N_t=2000, N_S=2000$) |
| **Iterative Method** | Integral operator Gaussian Picard | Product trapezoidal Volterra–Picard |
| **Learned Method** | Euler–Maruyama Neural SDE | Latent Volterra Neural SDE with power kernel |
| **Interpolation Space** | $S \in [60, 140], \sigma \in [0.1, 0.4], T \in [0.1, 2.0]$ | $\alpha \in [0.45, 0.95], S \in [70, 130], T \in [0.2, 1.5]$ |
| **Extrapolation Regime** | Deep OTM/ITM, High Vol ($\sigma \le 0.8$), Long $T$ | Sub-diffusive memory ($\alpha < 0.40$), Extreme $S$ |

---

## Deliverables Generated

Upon pipeline completion, the following artifacts are written to `outputs/`:

* `outputs/raw/block_a_results.csv`: Per-point errors, runtimes, and convergence logs for Block A.
* `outputs/raw/block_b_results.csv`: Per-point errors, runtimes, and convergence logs for Block B.
* `outputs/figures/convergence_rates.png`: Iteration-wise contraction ($\hat{L}_k$) vs. Neural SDE training loss.
* `outputs/figures/error_surfaces_bsm.png`: Pricing error surfaces across $(S, \sigma)$ for Picard vs. NSDE.
* `outputs/figures/fractional_alpha_slices.png`: Error evolution as a function of fractional order $\alpha$.
* `outputs/figures/failure_regimes.png`: Flagged violations (arbitrage bounds, monotonicity, NaN/Inf).
* `memo/research_memo.pdf`: Summary report detailing numerical findings, runtime tradeoffs, and failure modes.

---

## Reproducibility Protocol

* **Pre-inspection SHA:** Pin your repository commit prior to analyzing results:
```bash
git rev-parse HEAD

```


* **Exploratory Policy:** Any configuration tweaks or architectural tests conducted post-inspection must be committed on an `exploratory/*` branch and must not overwrite the baseline scripts in `solvers/` or configs in `configs/`.
