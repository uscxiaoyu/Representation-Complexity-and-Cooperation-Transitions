# Figure 3 Review Package

This folder is a self-contained review-stage reproduction package for
the manuscript Figure 3 training-dynamics panel. It is intentionally smaller
than `code-for-review`: it keeps only the ABM implementation, the three
state-representation DQN agents, the Figure 3 reproduction command, and a
small smoke test proving that SRC1, SRC2, and SRC3 ABM workflows execute.

Repository:
<https://github.com/uscxiaoyu/Representation-Complexity-and-Cooperation-Transitions>

## Contents

- `modules/model/adaptive_network_model.py`: adaptive-network ABM.
- `modules/model/dqn_agent_src1_vector.py`: SRC1 vector-state DQN.
- `modules/model/dqn_agent_src2_relational.py`: SRC2 relational-state DQN.
- `modules/model/dqn_agent_src3_structural_gnn.py`: SRC3 structural GNN DQN.
- `modules/reproduce_figure3_training_dynamics.py`: Figure 3 reproduction workflow.
- `modules/run_three_abm_smoke.py`: short SRC1/SRC2/SRC3 execution test.

No script imports the main repository or the larger `code-for-review` package.
The `outputs/` directory is not part of the distributed package. It is created
automatically on first run.

## Environment

```bash
cd <package-folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If platform-specific `torch-geometric` wheels are required, install the wheel
matching the local PyTorch and Python versions following the official PyTorch
Geometric instructions.

## Reproduce Figure 3

To reproduce Figure 3 from a package that does not include `outputs/`, run:

```bash
python scripts/reproduce_figure3.py --run-missing --workers 3
```

This command creates:

- `outputs/experiments/results/`
- `outputs/figures/figure3_reproduction/`
- `outputs/reports/`

and writes the three Figure 3 result files, the reproduced PNG, and a JSON
summary into those folders. The full run executes three ABM configurations and
may take substantial time depending on the reviewer machine.

If the three Figure 3 result files are already present under
`outputs/experiments/results/`, the figure can be regenerated directly:

```bash
python scripts/reproduce_figure3.py
```

The exact Figure 3 configuration is fixed in the script:
`SRC3`, `r_DRL=0.1`, `seed=0`, `max_memory=1`, `N=1000`, and
pre-training / training / testing epochs of `1000 / 3000 / 3000`.

## Verify the Three ABM Variants

```bash
python scripts/run_three_abm_smoke.py
```

This smoke test uses small runtime overrides and writes:

- `outputs/smoke_runs/quick_abm_smoke_summary_seed0.json`

It is not manuscript evidence; it is a workflow check showing that SRC1, SRC2,
and SRC3 all complete pre-training, training, and testing phases under the same
ABM implementation.
