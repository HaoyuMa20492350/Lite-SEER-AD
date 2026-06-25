from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from seer_ad_v2.utils.io import save_json


def _run_text(command: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        text = result.stdout.strip() or result.stderr.strip()
        return text if text else "unavailable"
    except Exception as exc:
        return f"unavailable: {exc}"


def git_hash(cwd: str | Path = ".") -> str:
    return _run_text(["git", "rev-parse", "HEAD"], Path(cwd))


def environment_lines(device: str) -> list[str]:
    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"device={device}",
    ]
    try:
        import torch

        lines.append(f"torch={torch.__version__}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_version={torch.version.cuda}")
            lines.append(f"gpu_count={torch.cuda.device_count()}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                mem_gb = props.total_memory / (1024 ** 3)
                lines.append(f"gpu_{idx}={props.name}, memory_gb={mem_gb:.2f}")
    except Exception as exc:
        lines.append(f"torch_info_error={exc}")
    return lines


def save_run_metadata(
    run_dir: str | Path,
    cfg: dict[str, Any],
    args: argparse.Namespace | dict[str, Any],
    device: str,
    command_name: str,
) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    with (run_path / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    args_dict = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    save_json({"command": command_name, "args": args_dict}, run_path / "run_args.json")
    (run_path / "environment.txt").write_text("\n".join(environment_lines(device)) + "\n", encoding="utf-8")
    (run_path / "git_hash.txt").write_text(git_hash(Path.cwd()) + "\n", encoding="utf-8")
