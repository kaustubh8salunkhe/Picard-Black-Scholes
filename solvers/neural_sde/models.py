import torch
import torch.nn as nn


class DiffusionMLP(nn.Module):
    """Pinned 3-layer MLP diffusion surrogate for Block A."""
    def __init__(self, in_features=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.net(x)) + 1e-4


class VolterraDriftDiffusionMLP(nn.Module):
    """Pinned Drift and Diffusion MLPs for Block B Fractional NSDE."""
    def __init__(self, in_features=3, hidden_dim=64):
        super().__init__()
        self.drift = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.diffusion = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, t_x_alpha):
        b = self.drift(t_x_alpha)
        sig = self.softplus(self.diffusion(t_x_alpha)) + 1e-4
        return b, sig
