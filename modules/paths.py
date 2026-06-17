"""Path management for the Figure 3 reproduction package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for the review package."""

    repo_root: Path
    package_dir: Path
    scripts_dir: Path
    outputs_dir: Path
    experiments_dir: Path
    results_dir: Path
    figures_dir: Path
    reports_dir: Path

    def ensure_output_dirs(self) -> None:
        """Create the output directory layout."""
        for directory in (
            self.outputs_dir,
            self.experiments_dir,
            self.results_dir,
            self.figures_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _build_project_paths() -> ProjectPaths:
    """Build project paths relative to this package location."""
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent
    outputs_dir = repo_root / "outputs"
    experiments_dir = outputs_dir / "experiments"
    return ProjectPaths(
        repo_root=repo_root,
        package_dir=package_dir,
        scripts_dir=repo_root / "scripts",
        outputs_dir=outputs_dir,
        experiments_dir=experiments_dir,
        results_dir=experiments_dir / "results",
        figures_dir=outputs_dir / "figures",
        reports_dir=outputs_dir / "reports",
    )


PROJECT_PATHS = _build_project_paths()
