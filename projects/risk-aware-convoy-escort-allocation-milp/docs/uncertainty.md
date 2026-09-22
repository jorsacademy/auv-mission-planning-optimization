# Uncertainty-Aware Extension

The deterministic model assumes that every convoy threat score is known when escort assignments are selected. This extension replaces that assumption with a finite set of synthetic threat scenarios.

All data in this repository are fictional. The uncertainty model is an operations-research example, not an empirical threat model or operational decision system.

## Scenario representation

Let `S` be the set of scenarios. Each scenario `s` has:

- probability `p[s]`,
- a positive threat multiplier `m[s,c]` for every convoy `c`.

The realized threat for convoy `c` in scenario `s` is:

`threat[c] * m[s,c]`

Scenario probabilities must be strictly positive and sum to one.

## First-stage decisions

Escort assignments are selected before the scenario is known. Therefore the assignment variables are shared by every scenario:

- `x[c,e] = 1` if escort `e` is assigned to convoy `c`,
- `z[c,e,k] = 1` if escort `e` occupies marginal-effect slot `k` for convoy `c`,
- `u[c,k] = 1` if slot `k` is active for convoy `c`.

These are the same structural decisions used by the deterministic model.

## Scenario-dependent gain

For scenario `s`, convoy `c`, escort `e`, and slot `k`, the uncapped survival increment is:

`scale * protection[e] * return_factor[k] / (threat[c] * m[s,c])`

The coefficient is capped by the remaining survival probability of the convoy. A scenario-specific feasibility constraint ensures that the selected gains cannot make modeled survival probability exceed one.

## Scenario outcome

The total expected surviving ships in scenario `s` is:

`Q[s] = baseline_expected_survivors + sum(c,e,k) ships[c] * gain[s,c,e,k] * z[c,e,k]`

The probability-weighted stochastic objective component is:

`E[Q] = sum(s) p[s] * Q[s]`

## Risk-neutral stochastic optimization

With `risk_aversion = 0`, the model solves:

`maximize E[Q]`

This selects a single escort allocation that maximizes expected survivors across the supplied scenario distribution.

## Finite-scenario robust optimization

Introduce a continuous variable `eta` constrained by:

`eta <= Q[s]` for every scenario `s`.

Then `eta` is the worst-case scenario outcome for the selected assignment.

With `risk_aversion = 1`, the model solves:

`maximize eta`

This is a maximin robust allocation over the finite scenario set. It does not assume an ambiguity set beyond the explicitly supplied scenarios.

## Risk-adjusted objective

For `risk_aversion = alpha` with `0 <= alpha <= 1`, the implemented objective is:

`maximize (1 - alpha) * E[Q] + alpha * eta`

Interpretation:

- `alpha = 0`: risk-neutral expected-value solution,
- `0 < alpha < 1`: compromise between average and worst-case performance,
- `alpha = 1`: finite-scenario maximin solution.

This weighted objective is linear and remains a MILP.

## Important interpretation limits

The model is not a two-stage recourse model because escort assignments cannot change after a scenario is observed. It is more accurately described as a scenario-based here-and-now stochastic MILP with an optional robust maximin term.

The scenario probabilities and threat multipliers are modeling assumptions. Results are conditional on those assumptions and should not be interpreted as calibrated forecasts.
