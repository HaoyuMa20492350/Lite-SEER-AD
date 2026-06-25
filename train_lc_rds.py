from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from seer_ad_v2.config import cfg_device, cfg_seed, latency_budget_ms, load_config, make_run_dir, resolve_device
from seer_ad_v2.models.scheduler.lc_rds import ACTION_NAMES, LCRDS, RuleScheduler
from seer_ad_v2.utils.io import save_checkpoint, save_json
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LC-RDS from a rule-teacher initializer.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--run-name", default="default")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg_seed(cfg, args.seed)
    seed_everything(seed)
    device = resolve_device(cfg_device(cfg, args.device))
    run_dir = make_run_dir(cfg, args.run_name)
    save_run_metadata(run_dir, cfg, args, device, "train_lc_rds")
    rng = np.random.default_rng(seed)
    x = rng.random((args.samples, 8), dtype=np.float32)
    rule = RuleScheduler(latency_budget_ms=latency_budget_ms(cfg))
    y_actions = [rule.choose(row).name for row in x]
    action_to_idx = {s: i for i, s in enumerate(ACTION_NAMES)}
    y = torch.tensor([action_to_idx[s] for s in y_actions], dtype=torch.long, device=device)
    x_t = torch.from_numpy(x).float().to(device)
    model = LCRDS().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(args.epochs):
        logits = model(x_t)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    save_checkpoint(run_dir / "lc_rds.pt", model_state=model.state_dict(), cfg=cfg, action_names=ACTION_NAMES)
    save_json({"loss": losses, "teacher": "RuleScheduler", "samples": args.samples}, run_dir / "lc_rds_metrics.json")
    print(f"Saved LC-RDS checkpoint to {run_dir / 'lc_rds.pt'}")


if __name__ == "__main__":
    main()
