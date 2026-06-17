"""Quick smoke test for the three ABM state-representation variants."""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modules.paths import PROJECT_PATHS

if TYPE_CHECKING:
    from modules.model.adaptive_network_model import CooperationABM


SRC_LEVELS = {
    "SRC1": ("ic_1", "simple"),
    "SRC2": ("ic_2", "medium"),
    "SRC3": ("ic_3", "complex"),
}


def display_path(path: Path) -> str:
    """Return a package-relative path when possible."""
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_PATHS.repo_root.resolve()))
    except ValueError:
        return str(path)


def load_model_api() -> tuple[Any, Any, Any, Any]:
    """Load the ABM implementation only when a smoke run is requested."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="An issue occurred while importing 'pyg-lib'.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="An issue occurred while importing 'torch-sparse'.*",
            category=UserWarning,
        )
        from modules.model.adaptive_network_model import (
            CooperationABM,
            get_experiment_configs,
            run_simulation,
            set_seed,
        )

    return CooperationABM, get_experiment_configs, run_simulation, set_seed


def build_smoke_config(args: argparse.Namespace, src_label: str) -> dict[str, Any]:
    """Build one reduced-size ABM configuration from the formal parameter grid."""
    _, get_experiment_configs, _, _ = load_model_api()
    input_key, input_complexity = SRC_LEVELS[src_label]
    config_key = f"{args.environment}&{input_key}"
    config = deepcopy(get_experiment_configs()[config_key])
    dqn_params = deepcopy(config["dqn_agent_params"])
    dqn_params.update(
        {
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "target_update_freq": args.target_update_freq,
        }
    )
    config.update(
        {
            "n": args.n_agents,
            "drl_agent_ratio": args.drl_agent_ratio,
            "pre_train_epochs": args.pre_train_epochs,
            "train_epochs": args.train_epochs,
            "test_epochs": args.test_epochs,
            "dqn_agent_params": dqn_params,
            "save_checkpoints": False,
        }
    )
    if input_complexity == "complex":
        config["max_memory"] = args.max_memory
    return config


def finite_last(values: list[float] | np.ndarray) -> float:
    """Return the last value after checking that the series is finite."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("The checked time series is empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("The checked time series contains non-finite values.")
    return float(array[-1])


def summarize_result(
    src_label: str,
    input_complexity: str,
    config: dict[str, Any],
    result: dict[str, Any],
    model: CooperationABM,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Summarize the core evidence that one ABM workflow completed."""
    pre_train = result["pre_train"]
    train_all = result["train"]["all"]
    test_all = result["test"]["all"]
    summary = {
        "src_label": src_label,
        "input_complexity": input_complexity,
        "status": "passed",
        "runtime_seconds": round(runtime_seconds, 3),
        "n_agents": config["n"],
        "drl_agent_ratio": config["drl_agent_ratio"],
        "phase_lengths": {
            "pre_train": len(pre_train["cooperation_rate"]),
            "train": len(train_all["cooperation_rate"]),
            "test": len(test_all["cooperation_rate"]),
        },
        "final_metrics": {
            "pre_train_cooperation_rate": finite_last(pre_train["cooperation_rate"]),
            "train_cooperation_rate": finite_last(train_all["cooperation_rate"]),
            "test_cooperation_rate": finite_last(test_all["cooperation_rate"]),
            "test_average_payoff": finite_last(test_all["payoffs"]),
            "test_average_degree": finite_last(test_all["avg_degree"]),
        },
        "drl_agent_evidence": {
            "replay_buffer_size": len(model.drl_agent.memory),
            "training_steps": int(getattr(model.drl_agent, "steps_done", 0)),
            "epsilon": float(getattr(model.drl_agent, "epsilon", 0.0)),
        },
    }
    expected = {
        "pre_train": config["pre_train_epochs"],
        "train": config["train_epochs"],
        "test": config["test_epochs"],
    }
    if summary["phase_lengths"] != expected:
        raise ValueError(f"Unexpected phase lengths for {src_label}: {summary}")
    return summary


def run_one_smoke_case(args: argparse.Namespace, src_label: str) -> dict[str, Any]:
    """Run one SRC-specific ABM smoke case."""
    CooperationABM, _, run_simulation, set_seed = load_model_api()
    input_key, input_complexity = SRC_LEVELS[src_label]
    config = build_smoke_config(args, src_label)
    set_seed(args.seed)
    print(
        f"Running {src_label} ({input_complexity}) using {args.environment}&{input_key}."
    )
    started_at = time.perf_counter()
    model = CooperationABM(**config)
    result = run_simulation(model)
    runtime_seconds = time.perf_counter() - started_at
    summary = summarize_result(
        src_label, input_complexity, config, result, model, runtime_seconds
    )
    if args.save_full_results:
        result_path = (
            args.output_dir / f"quick_abm_{src_label.lower()}_seed{args.seed}.pkl"
        )
        with open(result_path, "wb") as output_handle:
            pickle.dump(result, output_handle)
        summary["full_result_file"] = display_path(result_path)
    print(
        f"Completed {src_label}: test cooperation="
        f"{summary['final_metrics']['test_cooperation_rate']:.3f}, "
        f"training steps={summary['drl_agent_evidence']['training_steps']}."
    )
    return summary


def write_summary(output_dir: Path, seed: int, summaries: list[dict[str, Any]]) -> Path:
    """Write the smoke-test summary as reviewer-readable JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"quick_abm_smoke_summary_seed{seed}.json"
    payload = {
        "purpose": (
            "Quick reproducibility smoke test showing that SRC1, SRC2, and SRC3 "
            "ABM workflows all execute pre-training, training, and testing phases."
        ),
        "scope_note": (
            "This smoke test uses the formal model parameterization with explicit "
            "small-scale runtime overrides. It is not a substitute for the formal "
            "manuscript experiments."
        ),
        "summaries": summaries,
        "all_cases_passed": all(item["status"] == "passed" for item in summaries),
    }
    with open(output_file, "w", encoding="utf-8") as output_handle:
        json.dump(payload, output_handle, indent=2)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the ABM smoke test."""
    parser = argparse.ArgumentParser(
        description="Run a quick smoke test for SRC1, SRC2, and SRC3 ABM workflows."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--environment",
        choices=["ec_1", "ec_2", "ec_3"],
        default="ec_1",
        help="Payoff-environment configuration shared by the three SRC cases.",
    )
    parser.add_argument(
        "--n-agents", type=int, default=48, help="Smoke-test agent count."
    )
    parser.add_argument(
        "--drl-agent-ratio",
        type=float,
        default=0.25,
        help="Share of agents controlled by DRL policies in the smoke test.",
    )
    parser.add_argument("--pre-train-epochs", type=int, default=2)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--test-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--target-update-freq", type=int, default=2)
    parser.add_argument("--max-memory", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_PATHS.outputs_dir / "smoke_runs",
        help="Directory for the smoke-test JSON summary.",
    )
    parser.add_argument(
        "--save-full-results",
        action="store_true",
        help="Also save compact pickle files for the three smoke-test runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run all three ABM smoke cases."""
    args = build_parser().parse_args(argv)
    PROJECT_PATHS.ensure_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_one_smoke_case(args, src_label) for src_label in SRC_LEVELS]
    output_file = write_summary(args.output_dir, args.seed, summaries)
    print("All three ABM smoke cases completed successfully.")
    print(f"Summary written to: {display_path(output_file)}")


if __name__ == "__main__":
    main()
