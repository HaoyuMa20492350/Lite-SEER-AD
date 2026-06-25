from __future__ import annotations

import torch
import torch.nn.functional as F


def handcrafted_features(patches: torch.Tensor) -> torch.Tensor:
    """Cheap normal prototype features: RGB mean/std plus gradient magnitude stats."""
    mean = patches.mean(dim=(-2, -1))
    std = patches.std(dim=(-2, -1))
    gray = patches.mean(dim=1, keepdim=True)
    gx = gray[..., :, 1:] - gray[..., :, :-1]
    gy = gray[..., 1:, :] - gray[..., :-1, :]
    grad_mean = torch.stack([gx.abs().mean(dim=(-2, -1)), gy.abs().mean(dim=(-2, -1))], dim=1).squeeze(-1)
    feat = torch.cat([mean, std, grad_mean], dim=1)
    return F.normalize(feat, dim=1)


class PrototypeBank:
    def __init__(self) -> None:
        self.features: torch.Tensor | None = None
        self.distance_q50: torch.Tensor | None = None
        self.distance_q90: torch.Tensor | None = None
        self.distance_q95: torch.Tensor | None = None

    def fit(self, patches: torch.Tensor) -> None:
        self.features = handcrafted_features(patches).detach().cpu()
        self._calibrate()

    def append(self, patches: torch.Tensor) -> None:
        feats = handcrafted_features(patches).detach().cpu()
        self.features = feats if self.features is None else torch.cat([self.features, feats], dim=0)
        self._calibrate()

    def _calibrate(self) -> None:
        if self.features is None or len(self.features) < 2:
            self.distance_q50 = torch.tensor(0.0)
            self.distance_q90 = torch.tensor(0.0)
            self.distance_q95 = torch.tensor(1.0)
            return
        dists = torch.cdist(self.features, self.features, p=2)
        dists.fill_diagonal_(float("inf"))
        nearest = dists.min(dim=1).values
        self.distance_q50 = torch.quantile(nearest, 0.50).detach().cpu()
        self.distance_q90 = torch.quantile(nearest, 0.90).detach().cpu()
        self.distance_q95 = torch.quantile(nearest, 0.95).detach().cpu()

    def distance(self, patches: torch.Tensor) -> torch.Tensor:
        if self.features is None or len(self.features) == 0:
            return torch.zeros(patches.shape[0], device=patches.device)
        feats = handcrafted_features(patches)
        bank = self.features.to(patches.device)
        d = torch.cdist(feats, bank, p=2)
        return d.min(dim=1).values

    def novelty(self, patches: torch.Tensor) -> torch.Tensor:
        distance = self.distance(patches)
        if self.distance_q90 is None or self.distance_q95 is None or self.distance_q50 is None:
            self._calibrate()
        q50 = (self.distance_q50 if self.distance_q50 is not None else torch.tensor(0.0)).to(patches.device)
        q90 = (self.distance_q90 if self.distance_q90 is not None else torch.tensor(0.0)).to(patches.device)
        q95 = (self.distance_q95 if self.distance_q95 is not None else torch.tensor(1.0)).to(patches.device)
        scale = torch.clamp(q95 - q50, min=1e-3)
        return torch.sigmoid((distance - q90) / scale)

    def state_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "features": self.features,
            "distance_q50": self.distance_q50,
            "distance_q90": self.distance_q90,
            "distance_q95": self.distance_q95,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | None]) -> None:
        self.features = state.get("features")
        self.distance_q50 = state.get("distance_q50")
        self.distance_q90 = state.get("distance_q90")
        self.distance_q95 = state.get("distance_q95")
        if self.features is not None and (self.distance_q50 is None or self.distance_q90 is None or self.distance_q95 is None):
            self._calibrate()
