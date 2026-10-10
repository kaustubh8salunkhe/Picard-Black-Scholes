import argparse
import os
import yaml
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from solvers.neural_sde.models import DiffusionMLP, VolterraDriftDiffusionMLP
from solvers.neural_sde.engine import EulerMaruyamaPricer, FractionalVolterraPricer
from solvers.reference.analytical_bsm import bsm_call_price
from solvers.reference.l1_fdm_tfbsm import solve_tfbsm_l1


def train_block_a(cfg, device):
    torch.manual_seed(cfg["seeds"]["training"])
    train_df = pd.read_csv(f"data/datasets/{cfg['block_name']}_train.csv")

    targets = bsm_call_price(
        train_df["S"].values, train_df["K"].values,
        train_df["T"].values, train_df["r"].values, train_df["sigma"].values
    )

    dataset = TensorDataset(
        torch.tensor(train_df["S"].values, dtype=torch.float32),
        torch.tensor(train_df["K"].values, dtype=torch.float32),
        torch.tensor(train_df["T"].values, dtype=torch.float32),
        torch.tensor(train_df["r"].values, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=cfg["neural_sde"]["batch_size"], shuffle=True)

    model = DiffusionMLP(in_features=2, hidden_dim=cfg["neural_sde"]["hidden_units"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["neural_sde"]["learning_rate"]),
        weight_decay=float(cfg["neural_sde"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["neural_sde"]["epochs"],
        eta_min=float(cfg["neural_sde"]["min_learning_rate"])
    )

    model.train()
    for _ in range(cfg["neural_sde"]["epochs"]):
        for s_b, k_b, t_b, r_b, y_b in loader:
            optimizer.zero_grad()
            preds = EulerMaruyamaPricer.price_batch(
                model, s_b, k_b, t_b, r_b,
                steps=cfg["neural_sde"]["time_steps"],
                mc_paths=1000,
                device=device
            )
            loss = torch.mean((preds - y_b.to(device)) ** 2)
            loss.backward()
            optimizer.step()
        scheduler.step()

    os.makedirs("outputs/raw", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/raw/{cfg['block_name']}_model.pt")
    print(f"Saved trained weights to outputs/raw/{cfg['block_name']}_model.pt")


def train_block_b(cfg, device):
    torch.manual_seed(cfg["seeds"]["training"])
    train_df = pd.read_csv(f"data/datasets/{cfg['block_name']}_train.csv")

    # Generate reference targets for training set subset
    targets = []
    for _, row in train_df.iterrows():
        p = solve_tfbsm_l1(row["S"], row["K"], row["T"], row["r"], row["sigma"], row["alpha"], Nt=300, Ns=300)
        targets.append(p)

    dataset = TensorDataset(
        torch.tensor(train_df["alpha"].values, dtype=torch.float32),
        torch.tensor(train_df["S"].values, dtype=torch.float32),
        torch.tensor(train_df["K"].values, dtype=torch.float32),
        torch.tensor(train_df["T"].values, dtype=torch.float32),
        torch.tensor(train_df["r"].values, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=cfg["neural_sde"]["batch_size"], shuffle=True)

    model = VolterraDriftDiffusionMLP(in_features=3, hidden_dim=cfg["neural_sde"]["hidden_units"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["neural_sde"]["learning_rate"]),
        weight_decay=float(cfg["neural_sde"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["neural_sde"]["epochs"],
        eta_min=float(cfg["neural_sde"]["min_learning_rate"])
    )

    model.train()
    for _ in range(cfg["neural_sde"]["epochs"]):
        for a_b, s_b, k_b, t_b, r_b, y_b in loader:
            optimizer.zero_grad()
            preds = FractionalVolterraPricer.price_batch(
                model, a_b, s_b, k_b, t_b, r_b,
                steps=cfg["neural_sde"]["time_steps"],
                mc_paths=500,
                device=device
            )
            loss = torch.mean((preds - y_b.to(device)) ** 2)
            loss.backward()
            optimizer.step()
        scheduler.step()

    os.makedirs("outputs/raw", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/raw/{cfg['block_name']}_model.pt")
    print(f"Saved trained weights to outputs/raw/{cfg['block_name']}_model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "block_a" in cfg["block_name"]:
        train_block_a(cfg, device)
    else:
        train_block_b(cfg, device)


if __name__ == "__main__":
    main()
