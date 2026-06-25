from __future__ import annotations

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu") -> None:
        self.timesteps = timesteps
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    def loss(self, model: torch.nn.Module, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def reconstruct(self, model: torch.nn.Module, x0: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Few-step denoising reconstruction from a lightly noised input."""
        model.eval()
        if steps <= 0:
            return x0.clamp(-1, 1)
        steps = min(steps, self.timesteps - 1)
        start_t = max(1, int((steps / max(1, self.timesteps)) * (self.timesteps - 1)))
        start_t = max(start_t, min(self.timesteps - 1, steps * max(1, self.timesteps // 20)))
        t_start = torch.full((x0.shape[0],), start_t, device=x0.device, dtype=torch.long)
        x = self.q_sample(x0, t_start)
        schedule = torch.linspace(start_t, 0, steps + 1, device=x0.device).long()
        for i in range(len(schedule) - 1):
            t = schedule[i].repeat(x0.shape[0])
            t_next = schedule[i + 1]
            eps = model(x, t)
            alpha_t = self.alpha_bars[t].view(-1, 1, 1, 1)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * eps) / torch.sqrt(alpha_t).clamp_min(1e-8)
            alpha_next = self.alpha_bars[t_next].view(1, 1, 1, 1)
            x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * eps
        return x.clamp(-1, 1)
