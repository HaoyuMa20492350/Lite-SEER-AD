from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from seer_ad_v2.data.hard_negative_mining import ROI


@dataclass(frozen=True)
class SchedulerAction:
    name: str
    steps: int
    native_refine: bool = False


ACTION_SPECS = [
    SchedulerAction("skip", 0, False),
    SchedulerAction("repair-5", 5, False),
    SchedulerAction("repair-10", 10, False),
    SchedulerAction("repair-25", 25, False),
    SchedulerAction("native-refine", 40, True),
]
ACTION_NAMES = [a.name for a in ACTION_SPECS]
ACTION_STEPS = [a.steps for a in ACTION_SPECS]
_ACTION_BY_NAME = {a.name: a for a in ACTION_SPECS}
_ACTION_BY_STEP = {a.steps: a for a in ACTION_SPECS}
PRODUCTION_BUDGET_GUARD_LATENCY_MS = {
    "skip": 0.0,
    "repair-5": 80.0,
    "repair-10": 110.0,
    "repair-25": 140.0,
    "native-refine": 145.0,
}


def production_budget_guard_latency_estimates() -> dict[str, float]:
    return dict(PRODUCTION_BUDGET_GUARD_LATENCY_MS)


def action_from_name(name: str) -> SchedulerAction:
    if name in _ACTION_BY_NAME:
        return _ACTION_BY_NAME[name]
    if name.startswith("repair") and name.replace("repair", "").replace("-", "").isdigit():
        return SchedulerAction(name, int(name.replace("repair", "").replace("-", "")), False)
    raise ValueError(f"Unknown scheduler action: {name}")


def action_from_steps(steps: int) -> SchedulerAction:
    return _ACTION_BY_STEP.get(int(steps), SchedulerAction(f"repair-{int(steps)}", int(steps), False))


def roi_features(roi: ROI, sev_score: float, proto_distance: float, image_shape: tuple[int, int], roi_count: int) -> np.ndarray:
    h, w = image_shape
    area_ratio = roi.area / max(1.0, h * w)
    width = (roi.x2 - roi.x1) / max(1.0, w)
    height = (roi.y2 - roi.y1) / max(1.0, h)
    boundary = 2.0 * (width + height)
    return np.asarray([area_ratio, width, height, boundary, roi.peak, sev_score, proto_distance, roi_count], dtype=np.float32)


class LCRDS(nn.Module):
    def __init__(
        self,
        in_dim: int = 8,
        hidden: int = 32,
        num_actions: int = len(ACTION_NAMES),
        action_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.action_names = action_names or ACTION_NAMES[:num_actions]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RuleScheduler:
    def __init__(self, latency_budget_ms: float = 60.0, action_steps: list[int] | None = None) -> None:
        self.latency_budget_ms = latency_budget_ms
        self.action_steps = action_steps or ACTION_STEPS

    def choose(self, features: np.ndarray, spent_ms: float = 0.0) -> SchedulerAction:
        area, _, _, boundary, peak, sev, proto, roi_count = features
        value = 0.45 * sev + 0.2 * peak + 0.2 * proto + 0.15 * min(1.0, boundary)
        if spent_ms >= self.latency_budget_ms:
            return action_from_name("skip")
        if value > 0.72 and area < 0.35:
            return action_from_steps(25)
        if value > 0.48:
            return action_from_steps(10)
        return action_from_name("skip")


class ExpectedUtilityScheduler:
    def __init__(
        self,
        latency_budget_ms: float = 60.0,
        ema_decay: float = 0.8,
        latency_estimates: dict[str, float] | None = None,
    ) -> None:
        self.latency_budget_ms = float(latency_budget_ms)
        self.ema_decay = float(np.clip(ema_decay, 0.0, 0.999))
        self.expected_latency = {
            "skip": 0.0,
            "repair-5": 20.0,
            "repair-10": 40.0,
            "repair-25": 75.0,
            "native-refine": 80.0,
        }
        if latency_estimates:
            self.expected_latency.update({name: max(0.0, float(value)) for name, value in latency_estimates.items()})
        self.gain_multiplier = {
            "skip": 0.0,
            "repair-5": 0.52,
            "repair-10": 0.80,
            "repair-25": 1.35,
            "native-refine": 1.75,
        }
        self.observation_count = {name: 0 for name in self.expected_latency}

    def observe(self, action: SchedulerAction, latency_ms: float, realized_gain: float, predicted_gain: float | None = None) -> None:
        if action.name == "skip":
            return
        decay = self.ema_decay
        latency = max(0.0, float(latency_ms))
        current_latency = self.expected_latency.get(action.name, latency)
        smoothed = decay * current_latency + (1.0 - decay) * latency
        self.expected_latency[action.name] = max(current_latency, smoothed, latency)
        if predicted_gain is not None and predicted_gain > 1e-8:
            ratio = float(np.clip(realized_gain / predicted_gain, 0.05, 4.0))
            current = self.gain_multiplier.get(action.name, 1.0)
            self.gain_multiplier[action.name] = decay * current + (1.0 - decay) * ratio
        self.observation_count[action.name] = self.observation_count.get(action.name, 0) + 1

    def choose(self, features: np.ndarray, spent_ms: float = 0.0, expected_gain: float | None = None) -> SchedulerAction:
        area, _, _, boundary, peak, sev, proto, roi_count = features
        gain = float(expected_gain) if expected_gain is not None else float(0.35 * sev + 0.25 * peak + 0.25 * proto + 0.15 * min(1.0, boundary))
        if spent_ms >= self.latency_budget_ms or gain < 0.15:
            return action_from_name("skip")
        strength = float(np.clip(max(gain, sev, proto, peak), 0.0, 1.5))
        candidates = [
            ("skip", 0.0),
            ("repair-5", gain * self.gain_multiplier["repair-5"] * (1.0 - 0.06 * area)),
            ("repair-10", gain * self.gain_multiplier["repair-10"] * (1.0 - 0.10 * area)),
            ("repair-25", gain * self.gain_multiplier["repair-25"] * (0.82 + 0.32 * strength - 0.18 * area)),
            (
                "native-refine",
                gain * self.gain_multiplier["native-refine"] * (0.72 + 0.45 * strength - 0.25 * area)
                if area < 0.2
                else -1.0,
            ),
        ]
        remaining = max(0.0, self.latency_budget_ms - spent_ms)
        feasible = []
        for name, expected in candidates:
            latency = self.expected_latency[name]
            if latency <= remaining:
                utility = expected / max(latency, 1.0) if name != "skip" else 0.0
                feasible.append((utility, expected, name))
        if not feasible:
            return action_from_name("skip")
        utility, expected, name = max(feasible, key=lambda item: (item[0], item[1]))
        if utility <= 0.0:
            return action_from_name("skip")
        return action_from_name(name)


@torch.no_grad()
def choose_action_with_model(model: LCRDS | None, features: np.ndarray, device: str = "cpu") -> SchedulerAction:
    if model is None:
        return RuleScheduler().choose(features)
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    action_idx = int(model(x).argmax(dim=1).item())
    action_names = getattr(model, "action_names", ACTION_NAMES)
    if 0 <= action_idx < len(action_names):
        return action_from_name(action_names[action_idx])
    return ACTION_SPECS[action_idx]


@torch.no_grad()
def choose_with_model(model: LCRDS | None, features: np.ndarray, device: str = "cpu") -> int:
    return choose_action_with_model(model, features, device).steps
