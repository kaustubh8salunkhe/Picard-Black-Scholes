import torch


class EulerMaruyamaPricer:
    @staticmethod
    def price_batch(model, S, K, T, r, steps=100, mc_paths=10000, device="cpu"):
        S = S.to(device)
        K = K.to(device)
        T = T.to(device)
        r = r.to(device)

        batch_size = len(S)
        dt = (T / steps).unsqueeze(1)
        sqrt_dt = torch.sqrt(dt)

        X = S.unsqueeze(1).repeat(1, mc_paths)
        r_mat = r.unsqueeze(1).repeat(1, mc_paths)

        for step in range(steps):
            t_curr = (step * dt).repeat(1, mc_paths)
            inp = torch.stack([t_curr, X], dim=-1)
            sig = model(inp).squeeze(-1)
            dW = torch.randn_like(X) * sqrt_dt
            X = X + r_mat * X * dt + sig * X * dW

        payoff = torch.clamp(X - K.unsqueeze(1), min=0.0)
        discounted = torch.exp(-r.unsqueeze(1) * T.unsqueeze(1)) * torch.mean(payoff, dim=1)
        return discounted


class FractionalVolterraPricer:
    @staticmethod
    def price_batch(model, alpha, S, K, T, r, steps=128, mc_paths=5000, device="cpu"):
        S = S.to(device)
        K = K.to(device)
        T = T.to(device)
        r = r.to(device)
        alpha = alpha.to(device)

        dt = (T / steps).unsqueeze(1)
        sqrt_dt = torch.sqrt(dt)

        X = S.unsqueeze(1).repeat(1, mc_paths)
        alpha_mat = alpha.unsqueeze(1).repeat(1, mc_paths)

        for step in range(steps):
            t_curr = (step * dt).repeat(1, mc_paths)
            inp = torch.stack([t_curr, X, alpha_mat], dim=-1)
            b, sig = model(inp)
            b = b.squeeze(-1)
            sig = sig.squeeze(-1)

            kernel_weight = ((step + 1.0) ** (alpha.unsqueeze(1) - 0.5)) / torch.exp(torch.lgamma(alpha.unsqueeze(1) + 0.5))
            dW = torch.randn_like(X) * sqrt_dt
            X = X + b * dt + kernel_weight * sig * dW

        payoff = torch.clamp(X - K.unsqueeze(1), min=0.0)
        discounted = torch.exp(-r.unsqueeze(1) * T.unsqueeze(1)) * torch.mean(payoff, dim=1)
        return discounted
