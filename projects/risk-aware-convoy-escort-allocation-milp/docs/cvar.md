# Lower-Tail CVaR Extension

This extension adds Conditional Value at Risk (CVaR) to the finite-scenario allocation model.

The repository remains fully synthetic. CVaR is used here as a general operations-research risk measure for low-outcome scenarios. It is not a calibrated operational risk model.

## Why lower-tail CVaR

The stochastic model maximizes probability-weighted expected survivors. The maximin model maximizes the single worst scenario. CVaR provides an intermediate risk concept: it optimizes the average outcome in the lower tail of the scenario distribution rather than only the mean or only the single worst case.

Because larger survivor totals are desirable, this project uses lower-tail CVaR of survivors.

Let `Q[s]` be total expected survivors in scenario `s`, with probability `p[s]`. Let `alpha` be the CVaR confidence level, where:

`0 <= alpha < 1`.

The lower tail has total probability mass:

`1 - alpha`.

For example, `alpha = 0.90` focuses on the worst 10% probability mass.

## Linear formulation

Introduce a continuous threshold variable `eta` and nonnegative shortfall variables `d[s]`.

For every scenario:

`d[s] >= eta - Q[s]`

and:

`d[s] >= 0`.

The lower-tail CVaR expression is:

`CVaR_alpha(Q) = eta - (1 / (1 - alpha)) * sum(s) p[s] * d[s]`.

Maximizing this expression selects `eta` and the escort allocation so that the average outcome in the lower tail is as large as possible.

This is the reward-side equivalent of the standard Rockafellar-Uryasev linear CVaR construction. The sign is oriented for a quantity that should be maximized rather than a loss that should be minimized.

## Mean-CVaR objective

The implementation exposes `cvar_weight` in `[0, 1]`:

`maximize (1 - cvar_weight) * E[Q] + cvar_weight * CVaR_alpha(Q)`.

Interpretation:

- `cvar_weight = 0`: risk-neutral expected-value optimization,
- `0 < cvar_weight < 1`: mean-CVaR trade-off,
- `cvar_weight = 1`: pure lower-tail CVaR optimization.

At `cvar_alpha = 0`, lower-tail CVaR equals the full probability-weighted expectation.

As `cvar_alpha` increases, the tail becomes narrower and more concentrated on adverse scenarios. With a finite discrete scenario set, CVaR can include a fractional share of the scenario probability that crosses the tail boundary.

## CVaR versus maximin robustness

CVaR and maximin are related but distinct.

Maximin uses only the smallest scenario outcome:

`maximize min_s Q[s]`.

CVaR uses the probability-weighted average of a lower-tail region. Therefore it accounts for both severity and probability across multiple adverse scenarios.

This can be preferable when a planner wants downside protection without making the entire solution depend on a single extreme scenario.

## API

Use:

```python
from convoy_allocation.uncertainty import solve_cvar_allocation

result = solve_cvar_allocation(
    convoys,
    escorts,
    scenarios,
    max_available_escorts,
    cvar_alpha=0.90,
    cvar_weight=0.75,
)
```

The result exposes:

- `expected_survivors`,
- `worst_case_survivors`,
- `cvar_survivors`,
- scenario-level outcomes,
- the selected escort assignment,
- the optimized weighted objective value.

## Interpretation limits

The CVaR result is conditional on the supplied discrete scenario probabilities. It does not estimate scenario probabilities, generate uncertainty distributions, or provide distributionally robust guarantees outside the finite scenario set.

The model remains a here-and-now MILP: assignments are fixed before the scenario is observed and no recourse decisions are introduced.
