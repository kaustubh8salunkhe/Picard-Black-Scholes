import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_block_a(raw_dir, out_dir):
    df_in = pd.read_csv(os.path.join(raw_dir, "block_a_bsm_test_in_results.csv"))
    df_ext = pd.read_csv(os.path.join(raw_dir, "block_a_bsm_test_ext_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error vs Moneyness S/K
    df_in["moneyness"] = df_in["S"] / df_in["K"]
    axes[0].scatter(df_in["moneyness"], np.abs(df_in["picard_price"] - df_in["ref_price"]), alpha=0.5, label="Picard ($L_1$ Error)", c="blue")
    axes[0].scatter(df_in["moneyness"], np.abs(df_in["nsde_price"] - df_in["ref_price"]), alpha=0.5, label="NSDE ($L_1$ Error)", c="red")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Moneyness (S/K)")
    axes[0].set_ylabel("Absolute Error (Log Scale)")
    axes[0].set_title("Block A: In-Domain Absolute Error")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--")

    # Extrapolation Failure Surfaces
    df_ext["moneyness"] = df_ext["S"] / df_ext["K"]
    axes[1].scatter(df_ext["moneyness"], df_ext["nsde_price"] - df_ext["ref_price"], alpha=0.6, label="NSDE Error", c="crimson")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Moneyness (S/K)")
    axes[1].set_ylabel("Pricing Residual")
    axes[1].set_title("Block A: OOD Extrapolation Residuals")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "block_a_performance.png"), dpi=300)
    plt.close()


def plot_block_b(raw_dir, out_dir):
    df_in = pd.read_csv(os.path.join(raw_dir, "block_b_tfbsm_test_in_results.csv"))

    plt.figure(figsize=(8, 5))
    plt.scatter(df_in["alpha"], np.abs(df_in["picard_price"] - df_in["ref_price"]), c="blue", label="Volterra-Picard Error")
    plt.scatter(df_in["alpha"], np.abs(df_in["nsde_price"] - df_in["ref_price"]), c="red", label="Fractional NSDE Error")
    plt.yscale("log")
    plt.xlabel(r"Fractional Order $\alpha$")
    plt.ylabel("Absolute Pricing Error (Log Scale)")
    plt.title(r"Block B: Error Degradation vs Memory Order $\alpha$")
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "block_b_alpha_sensitivity.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=str, choices=["A", "B"], required=True)
    parser.add_argument("--raw_dir", type=str, default="outputs/raw")
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.block == "A":
        plot_block_a(args.raw_dir, args.out_dir)
    else:
        plot_block_b(args.raw_dir, args.out_dir)
    print(f"Generated figures in {args.out_dir}")


if __name__ == "__main__":
    main()
