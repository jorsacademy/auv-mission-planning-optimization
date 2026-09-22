# Mathematical Formulation

This document describes the optimization model implemented in `src/convoy_allocation/model.py`.

All data in this repository are synthetic. The model is an operations-research demonstration and is not a validated operational or historical model.

## Sets and indices

- `c in C`: convoys
- `e in E`: escorts
- `k in K`: ordered escort-effect slots

The number of slots is

`|K| = min(max_available_escorts, number_of_return_factors, |E|)`.

## Input parameters

For each convoy `c`:

- `N_c`: number of ships
- `T_c > 0`: threat score
- `B_c in [0, 1]`: baseline survival probability
- `L_c`: minimum number of escorts
- `U_c`: maximum number of escorts

For each escort `e`:

- `P_e > 0`: protection score

Global parameters:

- `M`: maximum number of escorts that may be used
- `R_k > 0`: diminishing-return factor for slot `k`, with `R_k >= R_(k+1)`
- `alpha > 0`: effectiveness scale

## Survival-gain coefficients

The uncapped survival-probability increment for assigning escort `e` to convoy `c` in slot `k` is

`raw_gain[c,e,k] = alpha * P_e * R_k / T_c`.

The coefficient used by the MILP is

`G[c,e,k] = min(raw_gain[c,e,k], 1 - B_c)`.

A convoy-level constraint prevents the sum of selected increments from exceeding the remaining probability mass `1 - B_c`.

## Decision variables

### Assignment variable

`x[c,e] in {0,1}`

`x[c,e] = 1` when escort `e` is assigned to convoy `c`.

### Escort-slot variable

`z[c,e,k] in {0,1}`

`z[c,e,k] = 1` when escort `e` occupies marginal-effect slot `k` for convoy `c`.

### Slot activation variable

`s[c,k] in {0,1}`

`s[c,k] = 1` when slot `k` is used by convoy `c`.

## Objective

The model maximizes expected surviving ships:

`maximize sum_c N_c * B_c + sum_c sum_e sum_k N_c * G[c,e,k] * z[c,e,k]`

The baseline term is constant but is retained so the solver objective equals the reported expected-survivor total.

## Constraints

### Escort exclusivity

Each escort can serve at most one convoy:

`sum_c x[c,e] <= 1    for all e`.

### Fleet resource limit

`sum_c sum_e x[c,e] <= M`.

### Convoy escort bounds

`L_c <= sum_e x[c,e] <= U_c    for all c`.

The implementation also caps `U_c` at the number of modeled slots.

### Assignment-slot consistency

Every assigned escort occupies exactly one slot in its convoy:

`sum_k z[c,e,k] = x[c,e]    for all c,e`.

Each active slot contains exactly one escort:

`sum_e z[c,e,k] = s[c,k]    for all c,k`.

The number of active slots equals the number of assigned escorts:

`sum_k s[c,k] = sum_e x[c,e]    for all c`.

### Ordered slots

Later diminishing-return slots cannot be activated before earlier slots:

`s[c,k] <= s[c,k-1]    for all c and k > 0`.

### Survival probability cap

`sum_e sum_k G[c,e,k] * z[c,e,k] <= 1 - B_c    for all c`.

Therefore the modeled survival probability never exceeds one.

## Why the slot formulation is used

If escort effectiveness were modeled as a purely additive linear benefit with no marginal decay, an optimal solution could concentrate resources whenever one convoy has the largest benefit coefficient. Ordered slots retain a MILP structure while representing diminishing marginal effectiveness.

Because escorts have heterogeneous protection scores, `z[c,e,k]` is required rather than a convoy-only slot variable: the optimization must decide both which escort is assigned and which marginal slot that escort occupies.

## Internal consistency

The implementation constructs `G[c,e,k]` once and uses exactly those coefficients in three places:

1. the objective function,
2. the survival-probability feasibility constraint,
3. post-solve result reconstruction.

This prevents the common modeling error where one formula is optimized and a different formula is used to report expected outcomes.

## Interpretation limits

The numerical survival function is deliberately synthetic. The protection score, threat score, scale parameter, and diminishing-return factors are abstract model parameters rather than empirically calibrated probabilities. Results should therefore be interpreted as internally consistent optimization outputs for the supplied synthetic assumptions, not as real-world survival estimates.
