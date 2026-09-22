# Risk-Aware Convoy Escort Allocation MILP

A compact operations-research project for allocating a limited set of escort resources across multiple synthetic convoys under heterogeneous risk, diminishing marginal returns, and uncertain threat conditions.

The repository is intentionally synthetic. All convoy identifiers, escort identifiers, parameters, scenarios, and numerical values are fictional and are provided only to demonstrate optimization modeling techniques.

## Models included

The repository contains four related formulations:

1. **Deterministic MILP** — assumes each convoy threat score is known.
2. **Risk-neutral stochastic MILP** — selects one allocation before a finite threat scenario is observed and maximizes probability-weighted expected survivors.
3. **Finite-scenario robust MILP** — selects one allocation that maximizes the worst scenario outcome among the supplied scenarios.
4. **Lower-tail CVaR MILP** — maximizes downside-tail performance across adverse scenarios while optionally retaining an expected-value component.

A continuous `risk_aversion` parameter interpolates between expected-value and maximin objectives. The CVaR model separately exposes `cvar_alpha` and `cvar_weight` for explicit tail-risk control.

## Deterministic model

A planner assigns a limited number of escorts to convoys. Each convoy has a ship count, a baseline survival probability, a threat score, and optional minimum and maximum escort requirements.

Each escort has a protection score. Escort effectiveness is adjusted by convoy threat and by an ordered diminishing-return schedule. The model maximizes total expected surviving ships while respecting resource and assignment constraints.

The principal binary variables are:

- `x[c,e]`: escort `e` is assigned to convoy `c`,
- `z[c,e,k]`: escort `e` occupies diminishing-return slot `k` for convoy `c`,
- `s[c,k]`: slot `k` is active for convoy `c`.

For convoy `c`, escort `e`, and slot `k`, the synthetic survival increment is based on:

`effectiveness_scale * protection[e] * return_factor[k] / threat[c]`

The coefficient is capped by the convoy's remaining probability mass, and a convoy-level linear constraint ensures modeled survival probability cannot exceed one.

See [`docs/formulation.md`](docs/formulation.md) for the deterministic formulation.

## Uncertainty-aware models

The uncertainty extension introduces a finite scenario set. Each scenario has a probability and one positive threat multiplier for every convoy.

Escort assignments are **here-and-now decisions**: the same assignment must be used in every scenario. Scenario-dependent threat multipliers change the survival contribution of each selected escort-slot pair.

If `Q[s]` denotes total expected survivors under scenario `s`, the risk-neutral model maximizes:

`sum_s probability[s] * Q[s]`

For robust optimization, a variable `eta` is constrained by:

`eta <= Q[s]` for every scenario `s`

so maximizing `eta` maximizes the worst supplied scenario outcome.

The weighted expected/worst-case objective is:

`(1 - alpha) * expected_survivors + alpha * worst_case_survivors`

where `alpha = risk_aversion`.

- `alpha = 0.0`: risk-neutral stochastic allocation,
- `0.0 < alpha < 1.0`: expected/worst-case compromise,
- `alpha = 1.0`: finite-scenario maximin robust allocation.

See [`docs/uncertainty.md`](docs/uncertainty.md) for the full uncertainty formulation.

## Lower-tail CVaR model

The CVaR formulation protects against poor outcomes without reducing risk analysis to a single worst scenario.

For confidence level `cvar_alpha`, the lower-tail CVaR is the probability-weighted mean survivor outcome in the worst `1 - cvar_alpha` probability mass.

The implementation uses the linear formulation:

`CVaR_alpha(Q) = eta - (1 / (1 - alpha)) * sum_s p[s] * d[s]`

with:

`d[s] >= eta - Q[s]`

and `d[s] >= 0`.

The mean-CVaR objective is:

`(1 - cvar_weight) * E[Q] + cvar_weight * CVaR_alpha(Q)`

where:

- `cvar_weight = 0.0`: risk-neutral expected-value optimization,
- `0.0 < cvar_weight < 1.0`: mean-CVaR trade-off,
- `cvar_weight = 1.0`: pure lower-tail CVaR optimization.

At `cvar_alpha = 0`, CVaR equals the full expectation. Higher confidence levels concentrate the risk measure on progressively worse parts of the scenario distribution.

See [`docs/cvar.md`](docs/cvar.md) for the detailed formulation and interpretation limits.

## Why this formulation is stronger than a naive allocation model

This project avoids several common modeling errors by:

- using the same survival-gain coefficients in the objective, feasibility constraints, and result reconstruction,
- representing heterogeneous escorts explicitly,
- introducing ordered marginal-effect slots for diminishing returns,
- preventing survival probability from exceeding `1.0`,
- validating resource limits and aggregate minimum requirements before solving,
- rejecting non-finite and invalid numeric inputs,
- checking solver status before results are returned,
- supporting finite threat scenarios with explicit probabilities,
- supporting expected-value, maximin, and lower-tail CVaR objectives,
- treating CVaR correctly as a lower-tail reward measure rather than a loss minimization problem,
- covering deterministic, stochastic, robust, and tail-risk invariants with automated tests.

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── docs/
│   ├── cvar.md
│   ├── formulation.md
│   └── uncertainty.md
├── src/convoy_allocation/
│   ├── __init__.py
│   ├── cli.py
│   ├── model.py
│   ├── scenario.py
│   ├── uncertainty.py
│   └── uncertainty_cli.py
├── tests/
│   ├── test_model.py
│   └── test_uncertainty.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Run the deterministic demonstration

```bash
convoy-allocation
```

or:

```bash
python -m convoy_allocation.cli
```

## Run the uncertainty and CVaR demonstration

```bash
convoy-allocation-uncertain
```

or:

```bash
python -m convoy_allocation.uncertainty_cli
```

The command demonstrates risk-neutral stochastic optimization, maximin robustness, and a mean-CVaR allocation on the same synthetic scenario set.

## Python API example

```python
from convoy_allocation.scenario import (
    build_demo_scenario,
    build_uncertain_demo_scenarios,
)
from convoy_allocation.uncertainty import solve_cvar_allocation

convoys, escorts, limit = build_demo_scenario()
scenarios = build_uncertain_demo_scenarios()

result = solve_cvar_allocation(
    convoys,
    escorts,
    scenarios,
    limit,
    cvar_alpha=0.90,
    cvar_weight=0.75,
)

print(result.expected_survivors)
print(result.cvar_survivors)
print(result.worst_case_survivors)
```

## Run the test suite

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The test suite verifies, among other properties:

- deterministic objective/reporting consistency,
- escort exclusivity,
- survival-probability bounds,
- convoy minimum and maximum assignment limits,
- diminishing marginal benefit across slots,
- stochastic objective consistency with probability-weighted outcomes,
- robust objective consistency with the worst supplied scenario,
- equivalence between the deterministic model and a one-scenario nominal uncertainty model,
- rejection of invalid scenario probability distributions,
- CVaR equivalence to expected value at `cvar_alpha = 0`,
- risk-neutral equivalence when `cvar_weight = 0`,
- CVaR bounds between worst-case and expected performance,
- concentration of higher-confidence CVaR on a no-better lower tail,
- validation of CVaR confidence and objective-weight parameters.

GitHub Actions runs the test suite on Python 3.10, 3.11, and 3.12 for pushes and pull requests targeting `main`.

## Model scope and interpretation

This repository is an educational operations-research example. It is not a historical reconstruction, tactical tool, empirical survival model, forecast, or validated operational decision system.

The protection scores, threat scores, survival probabilities, scenario probabilities, threat multipliers, scale parameter, and diminishing-return factors are synthetic assumptions. The optimization is mathematically conditional on those assumptions, but the numerical outputs should not be interpreted as estimates for any real-world operation.

## Design limitations

The current models do not include travel time, multi-period scheduling, escort endurance, route compatibility, correlated loss processes, convoy splitting or merging, dynamic reallocation after scenario observation, recourse decisions, distributionally robust ambiguity sets, or empirically calibrated nonlinear survival functions.

The CVaR model is finite-scenario CVaR. It does not infer a continuous probability distribution or provide guarantees outside the supplied scenario set.

## License

This project is source-available for non-commercial research, education, and personal study only. Commercial use is prohibited. It is not distributed under an OSI-approved open-source license.

See [`LICENSE`](LICENSE) for the full terms.
