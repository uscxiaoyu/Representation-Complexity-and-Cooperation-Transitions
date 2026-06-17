"""Reproduce the manuscript Figure 3 training-dynamics panel.

This module is intentionally narrow: it reproduces the SRC3, r_DRL=0.1,
seed-0 training-dynamics figure used in the manuscript. The original notebook
selected the SRC3 max_memory=1 files for this figure, so the same design choice
is made explicit here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from modules.paths import PROJECT_PATHS


FIGURE3_LEVELS = ("ec_1&ic_3", "ec_2&ic_3", "ec_3&ic_3")
FIGURE3_ENV_LEVELS = (1, 2, 3)
FIGURE3_INPUT_COMPLEXITY = 3
FIGURE3_DRL_RATIO = 0.1
FIGURE3_RATIO_TAG = "ratio01"
FIGURE3_SEED = 0
FIGURE3_MAX_MEMORY = 1
FIGURE3_PRE_TRAIN_EPOCHS = 1000
FIGURE3_TRAIN_EPOCHS = 3000
FIGURE3_TEST_EPOCHS = 3000
FIGURE3_N_AGENTS = 1000
FIGURE3_FILENAME = "training_dynamics_ic3_ratio0.1_seed0.png"
FIGURE3_SERIES_PATHS = (
    ("pre_train.cooperation_rate", ("pre_train", "cooperation_rate")),
    ("train.all.cooperation_rate", ("train", "all", "cooperation_rate")),
    ("train.drl.cooperation_rate", ("train", "drl", "cooperation_rate")),
    ("train.imitation.cooperation_rate", ("train", "imitation", "cooperation_rate")),
    ("test.all.cooperation_rate", ("test", "all", "cooperation_rate")),
    ("test.drl.cooperation_rate", ("test", "drl", "cooperation_rate")),
    ("test.imitation.cooperation_rate", ("test", "imitation", "cooperation_rate")),
)


def configure_figure_style() -> None:
    """Apply the same Matplotlib style used by the manuscript figure script."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 500,
            "savefig.dpi": 500,
        }
    )


def result_path_for_level(results_dir: Path, level: str) -> Path:
    """Return the canonical Figure 3 result path for one PE level."""
    return results_dir / f"{FIGURE3_RATIO_TAG}_{level}_mem1_seed{FIGURE3_SEED}.pkl"


def _load_pickle(path: Path) -> dict[str, Any]:
    """Load a saved ABM result file."""
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def _save_pickle(path: Path, data: dict[str, Any]) -> None:
    """Save one ABM result file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(data, file_obj)


def _run_and_save_level(level: str, results_dir_str: str, overwrite: bool) -> str:
    """Run one Figure 3 ABM configuration and save its result file."""
    from modules.model.adaptive_network_model import (
        CooperationABM,
        get_experiment_configs,
        run_simulation,
        set_seed,
    )

    results_dir = Path(results_dir_str)
    output_path = result_path_for_level(results_dir, level)
    if output_path.exists() and not overwrite:
        return str(output_path)

    configs = get_experiment_configs()
    if level not in configs:
        raise ValueError(f"Unknown Figure 3 level: {level}")

    config = configs[level].copy()
    config.update(
        {
            "n": FIGURE3_N_AGENTS,
            "drl_agent_ratio": FIGURE3_DRL_RATIO,
            "max_memory": FIGURE3_MAX_MEMORY,
            "pre_train_epochs": FIGURE3_PRE_TRAIN_EPOCHS,
            "train_epochs": FIGURE3_TRAIN_EPOCHS,
            "test_epochs": FIGURE3_TEST_EPOCHS,
        }
    )

    set_seed(FIGURE3_SEED)
    model = CooperationABM(**config)
    start_time = time.perf_counter()
    result = run_simulation(model)
    result["run_time"] = time.perf_counter() - start_time
    result["config"] = config
    result["level"] = level
    result["seed"] = FIGURE3_SEED
    result["figure3_reproduction"] = figure3_design()

    _save_pickle(output_path, result)
    return str(output_path)


def figure3_design() -> dict[str, Any]:
    """Return the fixed experimental design used for Figure 3 reproduction."""
    return {
        "levels": list(FIGURE3_LEVELS),
        "input_complexity": FIGURE3_INPUT_COMPLEXITY,
        "drl_agent_ratio": FIGURE3_DRL_RATIO,
        "seed": FIGURE3_SEED,
        "max_memory": FIGURE3_MAX_MEMORY,
        "n_agents": FIGURE3_N_AGENTS,
        "pre_train_epochs": FIGURE3_PRE_TRAIN_EPOCHS,
        "train_epochs": FIGURE3_TRAIN_EPOCHS,
        "test_epochs": FIGURE3_TEST_EPOCHS,
    }


def ensure_results(
    results_dir: Path,
    run_missing: bool,
    overwrite: bool,
    workers: int,
) -> dict[int, tuple[Path, dict[str, Any]]]:
    """Load or generate the three Figure 3 result files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {
        ec_level: result_path_for_level(results_dir, level)
        for ec_level, level in zip(FIGURE3_ENV_LEVELS, FIGURE3_LEVELS)
    }

    missing = [path for path in expected_paths.values() if overwrite or not path.exists()]
    if missing and not run_missing:
        missing_list = "\n".join(f"- {report_path(path)}" for path in missing)
        raise FileNotFoundError(
            "Figure 3 result files are missing. Re-run with --run-missing "
            f"to generate them:\n{missing_list}"
        )

    if missing:
        levels_to_run = [
            level
            for level in FIGURE3_LEVELS
            if overwrite or not result_path_for_level(results_dir, level).exists()
        ]
        run_figure3_levels(levels_to_run, results_dir, overwrite, workers)

    return {
        ec_level: (path, _load_pickle(path))
        for ec_level, path in expected_paths.items()
    }


def run_figure3_levels(
    levels: list[str],
    results_dir: Path,
    overwrite: bool,
    workers: int,
) -> None:
    """Run missing Figure 3 ABM configurations."""
    if workers <= 1 or len(levels) <= 1:
        for level in levels:
            _run_and_save_level(level, str(results_dir), overwrite)
        return

    max_workers = min(workers, len(levels))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_and_save_level, level, str(results_dir), overwrite): level
            for level in levels
        }
        for future in as_completed(futures):
            future.result()


def plot_figure3(
    results_by_ec: dict[int, tuple[Path, dict[str, Any]]],
    output_path: Path,
) -> Path:
    """Plot the Figure 3 training-dynamics panel."""
    configure_figure_style()
    colors = ["#0173B2", "#DE8F05", "#029E73"]
    markers = ("o", "s", "^")
    titles = ("(a)", "(b)", "(c)")
    fig, axes = plt.subplots(3, 1, figsize=(8, 5))

    for idx, ec_level in enumerate(FIGURE3_ENV_LEVELS):
        ax = axes[idx]
        data = results_by_ec[ec_level][1]
        plot_single_axis(
            ax=ax,
            data=data,
            title=f"{titles[idx]} PE{ec_level}&SRC{FIGURE3_INPUT_COMPLEXITY}",
            colors=colors,
            markers=markers,
            show_legend=idx == 0,
        )

    axes[-1].set_xlabel("Episodes", fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_single_axis(
    ax: plt.Axes,
    data: dict[str, Any],
    title: str,
    colors: list[str],
    markers: tuple[str, str, str],
    show_legend: bool,
) -> None:
    """Plot one PE panel in the Figure 3 training-dynamics layout."""
    pre_train = data["pre_train"]["cooperation_rate"]
    train_all = data["train"]["all"]["cooperation_rate"]
    train_drl = data["train"]["drl"]["cooperation_rate"]
    train_il = data["train"]["imitation"]["cooperation_rate"]
    test_all = data["test"]["all"]["cooperation_rate"]
    test_drl = data["test"]["drl"]["cooperation_rate"]
    test_il = data["test"]["imitation"]["cooperation_rate"]

    pre_len = len(pre_train)
    train_len = len(train_all)
    test_len = len(test_all)
    pre_x = np.arange(0, pre_len)
    train_x = np.arange(pre_len, pre_len + train_len)
    test_x = np.arange(pre_len + train_len, pre_len + train_len + test_len)

    draw_series(ax, pre_x, pre_train, colors[0], markers[0])
    draw_series(ax, train_x, train_all, colors[0], markers[0], "Overall")
    draw_series(ax, train_x, train_drl, colors[1], markers[1], "DRL-agent")
    draw_series(ax, train_x, train_il, colors[2], markers[2], "IL-agent")
    draw_series(ax, test_x, test_all, colors[0], markers[0])
    draw_series(ax, test_x, test_drl, colors[1], markers[1])
    draw_series(ax, test_x, test_il, colors[2], markers[2])

    ax.axvline(x=pre_len, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(
        x=pre_len + train_len,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    add_phase_labels(ax, pre_len, train_len, test_len)
    ax.set_ylabel("Coop. Rate", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlim([0, 7000])
    ax.set_ylim([0, 1.05])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, loc="upper right", fontsize=7)


def draw_series(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    marker: str,
    label: str | None = None,
) -> None:
    """Draw one cooperation-rate trajectory."""
    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=0.7,
        alpha=0.8,
        marker=marker,
        markersize=3,
        markevery=50,
        markerfacecolor="none",
        markeredgewidth=0.5,
        label=label,
    )


def add_phase_labels(ax: plt.Axes, pre_len: int, train_len: int, test_len: int) -> None:
    """Add phase labels to one Figure 3 panel."""
    labels = (
        ("Pre-training", pre_len / 2),
        ("Training", pre_len + train_len / 2),
        ("Testing", pre_len + train_len + test_len / 2),
    )
    for label, x_position in labels:
        ax.text(
            x_position,
            0.95,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )


def build_summary(
    results_by_ec: dict[int, tuple[Path, dict[str, Any]]],
    output_figure: Path,
    reference_figure: Path | None,
    reference_results_dir: Path | None,
) -> dict[str, Any]:
    """Build a machine-readable Figure 3 reproduction summary."""
    summary = {
        "figure": "Figure 3",
        "design": figure3_design(),
        "result_files": {},
        "final_500_episode_means": {},
        "output_figure": report_path(output_figure),
    }
    for ec_level, (path, data) in results_by_ec.items():
        summary["result_files"][f"PE{ec_level}"] = report_path(path)
        summary["final_500_episode_means"][f"PE{ec_level}"] = summarize_result(data)

    if reference_figure is not None:
        summary["reference_comparison"] = compare_reference_figure(
            output_figure, reference_figure
        )
    if reference_results_dir is not None:
        summary["reference_result_comparison"] = compare_reference_results(
            results_by_ec, reference_results_dir
        )
    return summary


def summarize_result(data: dict[str, Any], last_n: int = 500) -> dict[str, float]:
    """Summarize the final-window cooperation rates in one ABM result."""
    return {
        "pre_train_overall": final_mean(data["pre_train"]["cooperation_rate"], last_n),
        "train_overall": final_mean(data["train"]["all"]["cooperation_rate"], last_n),
        "test_overall": final_mean(data["test"]["all"]["cooperation_rate"], last_n),
        "test_drl": final_mean(data["test"]["drl"]["cooperation_rate"], last_n),
        "test_imitation": final_mean(
            data["test"]["imitation"]["cooperation_rate"], last_n
        ),
    }


def final_mean(values: np.ndarray, last_n: int) -> float:
    """Return the mean over the last n entries."""
    return float(np.mean(values[-last_n:]))


def compare_reference_figure(generated_figure: Path, reference_figure: Path) -> dict[str, Any]:
    """Compare the generated figure with an optional reference PNG."""
    if not reference_figure.exists():
        raise FileNotFoundError(f"Reference figure does not exist: {reference_figure}")

    generated_image = mpimg.imread(generated_figure)
    reference_image = mpimg.imread(reference_figure)
    comparison: dict[str, Any] = {
        "generated_figure": report_path(generated_figure),
        "reference_figure": report_path(reference_figure),
        "generated_sha256": sha256_file(generated_figure),
        "reference_sha256": sha256_file(reference_figure),
        "generated_shape": list(generated_image.shape),
        "reference_shape": list(reference_image.shape),
    }
    comparison["same_sha256"] = (
        comparison["generated_sha256"] == comparison["reference_sha256"]
    )

    if generated_image.shape == reference_image.shape:
        diff = np.abs(generated_image.astype(float) - reference_image.astype(float))
        comparison["max_abs_pixel_diff"] = float(np.max(diff))
        comparison["mean_abs_pixel_diff"] = float(np.mean(diff))
        comparison["pixel_identical"] = bool(np.array_equal(generated_image, reference_image))
    else:
        comparison["max_abs_pixel_diff"] = None
        comparison["mean_abs_pixel_diff"] = None
        comparison["pixel_identical"] = False

    return comparison


def compare_reference_results(
    results_by_ec: dict[int, tuple[Path, dict[str, Any]]],
    reference_results_dir: Path,
) -> dict[str, Any]:
    """Compare reproduced ABM trajectories with reference result files."""
    if not reference_results_dir.exists():
        raise FileNotFoundError(
            f"Reference results directory does not exist: {reference_results_dir}"
        )

    comparison: dict[str, Any] = {
        "reference_results_dir": report_path(reference_results_dir),
        "levels": {},
        "all_series_identical": True,
        "max_abs_diff_overall": 0.0,
    }
    for ec_level, level in zip(FIGURE3_ENV_LEVELS, FIGURE3_LEVELS):
        generated_path, generated_data = results_by_ec[ec_level]
        reference_path = result_path_for_level(reference_results_dir, level)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference result file is missing: {reference_path}")

        level_comparison = compare_single_result(
            generated_data=generated_data,
            reference_data=_load_pickle(reference_path),
            generated_path=generated_path,
            reference_path=reference_path,
        )
        comparison["levels"][f"PE{ec_level}"] = level_comparison
        comparison["all_series_identical"] = (
            comparison["all_series_identical"]
            and level_comparison["all_series_identical"]
        )
        comparison["max_abs_diff_overall"] = max(
            comparison["max_abs_diff_overall"],
            level_comparison["max_abs_diff"],
        )

    return comparison


def compare_single_result(
    generated_data: dict[str, Any],
    reference_data: dict[str, Any],
    generated_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    """Compare key cooperation-rate series for one PE level."""
    level_comparison: dict[str, Any] = {
        "generated_file": report_path(generated_path),
        "reference_file": report_path(reference_path),
        "series": {},
        "all_series_identical": True,
        "max_abs_diff": 0.0,
    }
    for label, keys in FIGURE3_SERIES_PATHS:
        generated_series = np.asarray(resolve_series(generated_data, keys))
        reference_series = np.asarray(resolve_series(reference_data, keys))
        series_comparison = compare_series(generated_series, reference_series)
        level_comparison["series"][label] = series_comparison
        level_comparison["all_series_identical"] = (
            level_comparison["all_series_identical"]
            and series_comparison["identical"]
        )
        level_comparison["max_abs_diff"] = max(
            level_comparison["max_abs_diff"],
            series_comparison["max_abs_diff"] or 0.0,
        )
    return level_comparison


def resolve_series(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Resolve a nested series from an ABM result dictionary."""
    value: Any = data
    for key in keys:
        value = value[key]
    return value


def compare_series(
    generated_series: np.ndarray,
    reference_series: np.ndarray,
) -> dict[str, Any]:
    """Compare two numeric series."""
    result: dict[str, Any] = {
        "generated_length": int(generated_series.size),
        "reference_length": int(reference_series.size),
        "same_length": generated_series.shape == reference_series.shape,
    }
    if generated_series.shape != reference_series.shape:
        result.update(
            {
                "identical": False,
                "max_abs_diff": None,
                "mean_abs_diff": None,
            }
        )
        return result

    diff = np.abs(generated_series.astype(float) - reference_series.astype(float))
    result.update(
        {
            "identical": bool(np.array_equal(generated_series, reference_series)),
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
        }
    )
    return result


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path(path: Path) -> str:
    """Return a reviewer-safe path string without local absolute prefixes."""
    if not path.is_absolute():
        return path.as_posix()

    try:
        return path.resolve().relative_to(PROJECT_PATHS.repo_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def write_summary(summary: dict[str, Any], summary_path: Path) -> Path:
    """Write a JSON reproduction summary."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def prepare_output_layout(
    results_dir: Path,
    output_figure: Path,
    summary_file: Path,
) -> None:
    """Create output directories for a first run on a reviewer machine."""
    PROJECT_PATHS.ensure_output_dirs()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Reproduce the manuscript Figure 3 training-dynamics panel."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_PATHS.results_dir,
        help="Directory containing or receiving the three Figure 3 result files.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=PROJECT_PATHS.figures_dir / "figure3_reproduction" / FIGURE3_FILENAME,
        help="Path for the reproduced Figure 3 PNG.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=PROJECT_PATHS.reports_dir / "figure3_reproduction_summary.json",
        help="Path for the JSON reproduction summary.",
    )
    parser.add_argument(
        "--reference-figure",
        type=Path,
        default=None,
        help="Optional reference PNG for hash and pixel-level comparison.",
    )
    parser.add_argument(
        "--reference-results-dir",
        type=Path,
        default=None,
        help="Optional directory containing reference ABM result files for series comparison.",
    )
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="Run the three full ABM configurations if result files are absent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite existing Figure 3 result files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers used only when running missing ABM files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for Figure 3 reproduction."""
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    prepare_output_layout(args.results_dir, args.output_figure, args.summary_file)

    results = ensure_results(
        results_dir=args.results_dir,
        run_missing=args.run_missing,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    output_figure = plot_figure3(results, args.output_figure)
    summary = build_summary(
        results,
        output_figure,
        args.reference_figure,
        args.reference_results_dir,
    )
    summary_path = write_summary(summary, args.summary_file)

    print(f"Figure 3 reproduced: {report_path(output_figure)}")
    print(f"Reproduction summary written: {report_path(summary_path)}")
    if "reference_comparison" in summary:
        comparison = summary["reference_comparison"]
        print(f"Reference SHA-256 match: {comparison['same_sha256']}")
        print(f"Pixel-identical: {comparison['pixel_identical']}")
    if "reference_result_comparison" in summary:
        comparison = summary["reference_result_comparison"]
        print(f"Reference result series identical: {comparison['all_series_identical']}")
        print(f"Maximum result-series difference: {comparison['max_abs_diff_overall']}")


if __name__ == "__main__":
    main()
