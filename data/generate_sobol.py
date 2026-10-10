import argparse
import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import qmc


def sample_sobol_uniform(bounds, n_samples, seed):
    sampler = qmc.Sobol(d=len(bounds), seed=seed)
    m = int(np.ceil(np.log2(n_samples)))
    sample = sampler.random_base2(m=m)[:n_samples]
    lower = [b[0] for b in bounds]
    upper = [b[1] for b in bounds]
    return qmc.scale(sample, lower, upper)


def main():
    parser = argparse.ArgumentParser(description="Generate Sobol Parameter Grids")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--out_dir", type=str, default="data/datasets")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    block = cfg["block_name"]
    base_seed = cfg["seeds"]["dataset"]

    if "block_a" in block:
        keys = ["S", "sigma", "T", "r"]
        bounds_interp = [cfg["domains"]["interpolation"][k] for k in keys]

        train = sample_sobol_uniform(bounds_interp, cfg["dataset_sizes"]["train"], base_seed)
        val = sample_sobol_uniform(bounds_interp, cfg["dataset_sizes"]["val"], base_seed + 1)
        test_in = sample_sobol_uniform(bounds_interp, cfg["dataset_sizes"]["test_in_domain"], base_seed + 2)

        # Extrapolation sampling
        n_ext = cfg["dataset_sizes"]["test_extrapolation"]
        bounds_ext_other = [
            cfg["domains"]["extrapolation"]["sigma"],
            cfg["domains"]["extrapolation"]["T"],
            cfg["domains"]["extrapolation"]["r"],
        ]
        other_params = sample_sobol_uniform(bounds_ext_other, n_ext, base_seed + 3)

        rng = np.random.RandomState(base_seed + 4)
        s_choices = rng.choice([0, 1], size=n_ext)
        s_ranges = cfg["domains"]["extrapolation"]["S"]
        s_vals = np.zeros(n_ext)
        for i, choice in enumerate(s_choices):
            s_vals[i] = rng.uniform(s_ranges[choice][0], s_ranges[choice][1])

        test_ext = np.column_stack([s_vals, other_params])

        splits = {"train": train, "val": val, "test_in": test_in, "test_ext": test_ext}
        for name, data in splits.items():
            df = pd.DataFrame(data, columns=keys)
            df["K"] = cfg["strike"]
            df.to_csv(os.path.join(args.out_dir, f"{block}_{name}.csv"), index=False)

    else:
        keys = ["alpha", "S", "sigma", "T", "r"]
        bounds_interp = [cfg["domains"]["interpolation"][k] for k in keys]

        train = sample_sobol_uniform(bounds_interp, cfg["dataset_sizes"]["train"], base_seed)
        val = sample_sobol_uniform(bounds_interp, cfg["dataset_sizes"]["val"], base_seed + 1)
        test_in = sample_sobol_uniform(bounds_interp, cfg["dataset_sizes"]["test_in_domain"], base_seed + 2)

        n_ext = cfg["dataset_sizes"]["test_extrapolation"]
        bounds_ext_other = [
            cfg["domains"]["extrapolation"]["alpha"],
            cfg["domains"]["extrapolation"]["sigma"],
            cfg["domains"]["extrapolation"]["T"],
            cfg["domains"]["extrapolation"]["r"],
        ]
        other_params = sample_sobol_uniform(bounds_ext_other, n_ext, base_seed + 3)

        rng = np.random.RandomState(base_seed + 4)
        s_choices = rng.choice([0, 1], size=n_ext)
        s_ranges = cfg["domains"]["extrapolation"]["S"]
        s_vals = np.zeros(n_ext)
        for i, choice in enumerate(s_choices):
            s_vals[i] = rng.uniform(s_ranges[choice][0], s_ranges[choice][1])

        test_ext = np.column_stack([other_params[:, 0], s_vals, other_params[:, 1:]])

        splits = {"train": train, "val": val, "test_in": test_in, "test_ext": test_ext}
        for name, data in splits.items():
            df = pd.DataFrame(data, columns=keys)
            df["K"] = cfg["strike"]
            df.to_csv(os.path.join(args.out_dir, f"{block}_{name}.csv"), index=False)

    print(f"Generated frozen datasets for {block} in {args.out_dir}")


if __name__ == "__main__":
    main()
