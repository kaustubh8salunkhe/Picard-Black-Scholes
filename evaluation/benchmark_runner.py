import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
from solvers.reference.analytical_bsm import bsm_call_price
from solvers.reference.l1_fdm_tfbsm import solve_tfbsm_l1
from solvers.picard.picard_bsm import PicardBSMSolver
from solvers.picard.picard_tfbsm import PicardTFBSMSolver
from solvers.neural_sde.models import DiffusionMLP, VolterraDriftDiffusionMLP
from solvers.neural_sde.engine import EulerMaruyamaPricer, FractionalVolterraPricer
from evaluation.metrics import compute_rmse, compute_mape, compute_max_ae, audit_failures
from evaluation.timing import measure_latency


def evaluate_block_a(cfg, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    picard = PicardBSMSolver(
        Nx=cfg["picard"]["spatial_grid_size"],
        Ntau=cfg["picard"]["temporal_steps"],
        tol=float(cfg["picard"]["tol"]),
        max_iter=cfg["picard"]["max_iter"]
    )

    model = DiffusionMLP(in_features=2, hidden_dim=cfg["neural_sde"]["hidden_units"]).to(device)
    model.load_state_dict(torch.load(f"outputs/raw/{cfg['block_name']}_model.pt", map_location=device))
    model.eval()

    for split in ["test_in", "test_ext"]:
        df = pd.read_csv(f"data/datasets/{cfg['block_name']}_{split}.csv")
        ref_prices = bsm_call_price(df["S"], df["K"], df["T"], df["r"], df["sigma"])

        p_prices, p_iters, p_contractions = [], [], []
        for _, row in df.iterrows():
            p, it, c = picard.solve(row["S"], row["K"], row["T"], row["r"], row["sigma"])
            p_prices.append(p)
            p_iters.append(it)
            p_contractions.append(c)

        with torch.no_grad():
            nsde_prices = EulerMaruyamaPricer.price_batch(
                model,
                torch.tensor(df["S"].values, dtype=torch.float32),
                torch.tensor(df["K"].values, dtype=torch.float32),
                torch.tensor(df["T"].values, dtype=torch.float32),
                torch.tensor(df["r"].values, dtype=torch.float32),
                steps=cfg["neural_sde"]["time_steps"],
                mc_paths=cfg["neural_sde"]["mc_paths"],
                device=device
            ).cpu().numpy()

        p_prices = np.array(p_prices)
        df_out = df.copy()
        df_out["ref_price"] = ref_prices
        df_out["picard_price"] = p_prices
        df_out["picard_iters"] = p_iters
        df_out["picard_contraction"] = p_contractions
        df_out["nsde_price"] = nsde_prices

        p_audit = audit_failures(p_prices, df["S"], df["K"], df["T"], df["r"])
        n_audit = audit_failures(nsde_prices, df["S"], df["K"], df["T"], df["r"])

        df_out["picard_arb_violation"] = p_audit["arbitrage_violation"]
        df_out["nsde_arb_violation"] = n_audit["arbitrage_violation"]

        df_out.to_csv(os.path.join(out_dir, f"{cfg['block_name']}_{split}_results.csv"), index=False)
        print(f"[Block A | {split}] Picard RMSE: {compute_rmse(p_prices, ref_prices):.6f} | NSDE RMSE: {compute_rmse(nsde_prices, ref_prices):.6f}")


def evaluate_block_b(cfg, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    picard = PicardTFBSMSolver(
        Nt=cfg["picard"]["Nt"],
        Ns=cfg["picard"]["Ns"],
        tol=float(cfg["picard"]["tol"]),
        max_iter=cfg["picard"]["max_iter"]
    )

    model = VolterraDriftDiffusionMLP(in_features=3, hidden_dim=cfg["neural_sde"]["hidden_units"]).to(device)
    model.load_state_dict(torch.load(f"outputs/raw/{cfg['block_name']}_model.pt", map_location=device))
    model.eval()

    for split in ["test_in", "test_ext"]:
        df = pd.read_csv(f"data/datasets/{cfg['block_name']}_{split}.csv").head(100)  # High-precision subset
        ref_prices = []
        for _, row in df.iterrows():
            ref = solve_tfbsm_l1(row["S"], row["K"], row["T"], row["r"], row["sigma"], row["alpha"], Nt=500, Ns=500)
            ref_prices.append(ref)
        ref_prices = np.array(ref_prices)

        p_prices, p_iters, p_contractions = [], [], []
        for _, row in df.iterrows():
            p, it, c = picard.solve(row["S"], row["K"], row["T"], row["r"], row["sigma"], row["alpha"])
            p_prices.append(p)
            p_iters.append(it)
            p_contractions.append(c)
        p_prices = np.array(p_prices)

        with torch.no_grad():
            nsde_prices = FractionalVolterraPricer.price_batch(
                model,
                torch.tensor(df["alpha"].values, dtype=torch.float32),
                torch.tensor(df["S"].values, dtype=torch.float32),
                torch.tensor(df["K"].values, dtype=torch.float32),
                torch.tensor(df["T"].values, dtype=torch.float32),
                torch.tensor(df["r"].values, dtype=torch.float32),
                steps=cfg["neural_sde"]["time_steps"],
                mc_paths=cfg["neural_sde"]["mc_paths"],
                device=device
            ).cpu().numpy()

        df_out = df.copy()
        df_out["ref_price"] = ref_prices
        df_out["picard_price"] = p_prices
        df_out["picard_iters"] = p_iters
        df_out["picard_contraction"] = p_contractions
        df_out["nsde_price"] = nsde_prices

        p_audit = audit_failures(p_prices, df["S"], df["K"], df["T"], df["r"])
        n_audit = audit_failures(nsde_prices, df["S"], df["K"], df["T"], df["r"])

        df_out["picard_arb_violation"] = p_audit["arbitrage_violation"]
        df_out["nsde_arb_violation"] = n_audit["arbitrage_violation"]

        df_out.to_csv(os.path.join(out_dir, f"{cfg['block_name']}_{split}_results.csv"), index=False)
        print(f"[Block B | {split}] Picard RMSE: {compute_rmse(p_prices, ref_prices):.6f} | NSDE RMSE: {compute_rmse(nsde_prices, ref_prices):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=str, choices=["A", "B"], required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/raw")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.block == "A":
        with open("configs/block_a_bsm.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        evaluate_block_a(cfg, args.out_dir)
    else:
        with open("configs/block_b_tfbsm.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        evaluate_block_b(cfg, args.out_dir)


if __name__ == "__main__":
    main()
