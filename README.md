# AUV Mission Planning Optimization

A mixed-integer linear programming (MILP) framework for planning missions for a heterogeneous fleet of Autonomous Underwater Vehicles (AUVs).

The project models task assignment, vehicle routing, scheduling, compatibility constraints, operating-time limits, ocean-current effects, and a weighted mission objective. It is intended for research, education, and non-commercial experimentation in operations research and autonomous systems.

> **License notice:** This repository is source-available, not OSI open source. Commercial use is prohibited unless separate written permission is obtained from the copyright holder. See [LICENSE](LICENSE).

## Problem model

Each survey task must be completed exactly once. An AUV can perform a task only when its sensor payload contains the required sensor and its maximum depth rating covers the task depth. Every used AUV departs from and returns to the depot. Flow conservation and time-sequencing constraints define valid mission routes, while an operating-time constraint represents battery endurance.

Travel time is vehicle-specific and incorporates a simplified directional ocean-current effect. Energy consumption is tracked separately as a relative mission-cost proxy. The default objective minimizes a weighted combination of mission completion time and travel energy while retaining a task-priority term.

## Improvements over the original prototype

- deterministic NumPy and Python random generation via a single seed;
- every task is assigned exactly once (`== 1` rather than `<= 1`);
- AUV-specific travel-time and energy tensors;
- explicit binary variables for whether an AUV is used;
- depot departure/return tied to vehicle utilization;
- compatibility feasibility validation before solving;
- time sequencing doubles as subtour elimination for positive-duration routes;
- data-derived Big-M instead of a fixed arbitrary value;
- decision variables retained directly rather than reconstructed from PuLP variable-name strings;
- package structure, automated tests, linting, and GitHub Actions CI.

## Installation

Requires Python 3.10+ and PuLP's CBC solver support.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Basic usage

```python
from auv_optimizer import AUVOptimizer

optimizer = AUVOptimizer(num_auvs=4, num_tasks=15, seed=42)
optimizer.build_optimization_model()
solution = optimizer.solve(msg=True)

if solution is not None:
    optimizer.print_solution_summary()
    figure = optimizer.visualize_solution()
    figure.show()
```

Synthetic instances can occasionally contain a task for which no generated AUV is compatible. In that case the optimizer raises a `ValueError` before constructing an infeasible MILP. For controlled experiments, provide or modify fleet/task data before building the model.

## Objective

With default weights, the objective is conceptually:

```text
0.5 * mission completion time
+ 0.3 * relative travel energy
- 0.2 * completed-task priority
```

Since every task is mandatory, the priority term is constant for a fixed task set. It remains in the implementation to preserve the original formulation and to make future optional-task variants straightforward. For mandatory-task experiments it may be set to zero without changing the assignment decision.

## Tests

```bash
pytest --cov=auv_optimizer --cov-report=term-missing
ruff check .
```

The test suite checks deterministic data generation, vehicle-specific travel metrics, exact-once task completion, depot route structure, incompatibility detection, and plotting.

GitHub Actions runs the suite on Python 3.10, 3.11, and 3.12.

## Model scope and limitations

This is an operations-research reference implementation, not a certified vehicle-control or maritime navigation system. The current model deliberately simplifies hydrodynamics, energy consumption, currents, communications, collision avoidance, uncertainty, localization error, bathymetry, weather, and dynamic replanning. Do not use its output as a sole basis for safety-critical or real-world vehicle operation.

Potential extensions include time windows, optional tasks with priority rewards, multiple depots, heterogeneous energy models, charging/docking, stochastic currents, robust optimization, obstacle/bathymetry constraints, and rolling-horizon replanning.

## License

Copyright © 2026 jorsacademy.

Use is restricted by the repository's custom **Non-Commercial Source License**. Commercial use, commercial deployment, paid services, SaaS/API use, incorporation into commercial products, and commercial consulting use are prohibited without a separate written commercial license. Read the complete [LICENSE](LICENSE) before using or redistributing the software.
