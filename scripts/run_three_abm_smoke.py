#!/usr/bin/env python3
"""Thin wrapper for the three-ABM smoke test."""

from _bootstrap import REPO_ROOT  # noqa: F401
from modules.run_three_abm_smoke import main


if __name__ == "__main__":
    main()
