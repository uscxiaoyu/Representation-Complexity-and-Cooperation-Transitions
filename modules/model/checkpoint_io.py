"""Checkpoint adapter for the review reproduction package.

The Figure 3 workflow disables checkpoints. This adapter exists only so the
ABM implementation remains self-contained when imported from this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    agent: Any,
    checkpoint_path: str | Path,
    round_num: int,
    experiment_id: str,
    additional_info: dict[str, Any] | None = None,
) -> None:
    """Save a compact DQN checkpoint when checkpointing is explicitly enabled."""
    payload = {
        "round_num": round_num,
        "experiment_id": experiment_id,
        "additional_info": additional_info or {},
    }
    if hasattr(agent, "policy_net"):
        payload["policy_state_dict"] = agent.policy_net.state_dict()
    if hasattr(agent, "target_net"):
        payload["target_state_dict"] = agent.target_net.state_dict()
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
