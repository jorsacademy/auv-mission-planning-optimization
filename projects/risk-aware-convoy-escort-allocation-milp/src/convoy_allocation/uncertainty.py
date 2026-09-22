from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

from .model import Convoy, Escort


@dataclass(frozen=True)
class ThreatScenario:
    """Synthetic threat scenario used by the uncertainty-aware model."""

    probability: float
    threat_multipliers: Mapping[str, float]


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: str
    probability: float
    expected_survivors: float
    survival_rate: float


@dataclass(frozen=True)
class UncertainAllocationResult:
    status: str
    objective_value: float
    risk_aversion: float
    assigned_escorts: Mapping[str, tuple[str, ...]]
    scenarios: tuple[ScenarioOutcome, ...]

    @property
    def expected_survivors(self) -> float:
        return sum(item.probability * item.expected_survivors for item in self.scenarios)

    @property
    def worst_case_survivors(self) -> float:
        return min(item.expected_survivors for item in self.scenarios)


@dataclass(frozen=True)
class CVaRAllocationResult:
    status: str
    objective_value: float
    cvar_alpha: float
    cvar_weight: float
    assigned_escorts: Mapping[str, tuple[str, ...]]
    scenarios: tuple[ScenarioOutcome, ...]

    @property
    def expected_survivors(self) -> float:
        return sum(item.probability * item.expected_survivors for item in self.scenarios)

    @property
    def worst_case_survivors(self) -> float:
        return min(item.expected_survivors for item in self.scenarios)

    @property
    def cvar_survivors(self) -> float:
        return _lower_tail_cvar(self.scenarios, self.cvar_alpha)


_EPS = 1e-9


def _finite(value_: float) -> bool:
    return math.isfinite(float(value_))


def _validate_uncertain_inputs(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    scenarios: Mapping[str, ThreatScenario],
    max_available_escorts: int,
    diminishing_returns: Sequence[float],
    effectiveness_scale: float,
    risk_aversion: float,
) -> None:
    if not convoys:
        raise ValueError("At least one convoy is required.")
    if not escorts:
        raise ValueError("At least one escort is required.")
    if not scenarios:
        raise ValueError("At least one threat scenario is required.")
    if not isinstance(max_available_escorts, int) or isinstance(max_available_escorts, bool):
        raise TypeError("max_available_escorts must be an integer.")
    if not 0 <= max_available_escorts <= len(escorts):
        raise ValueError("max_available_escorts must be between zero and the number of escorts.")
    if not diminishing_returns:
        raise ValueError("diminishing_returns must contain at least one value.")
    if any(not _finite(v) or v <= 0 for v in diminishing_returns):
        raise ValueError("diminishing_returns values must be finite and positive.")
    if any(a < b for a, b in zip(diminishing_returns, diminishing_returns[1:])):
        raise ValueError("diminishing_returns must be non-increasing.")
    if not _finite(effectiveness_scale) or effectiveness_scale <= 0:
        raise ValueError("effectiveness_scale must be finite and positive.")
    if not _finite(risk_aversion) or not 0 <= risk_aversion <= 1:
        raise ValueError("risk_aversion must be in [0, 1].")

    probability_sum = 0.0
    convoy_ids = set(convoys)
    for scenario_id, scenario in scenarios.items():
        if not scenario_id:
            raise ValueError("Scenario identifiers must be non-empty strings.")
        if not _finite(scenario.probability) or scenario.probability <= 0:
            raise ValueError(f"Scenario {scenario_id} probability must be finite and positive.")
        probability_sum += scenario.probability
        if set(scenario.threat_multipliers) != convoy_ids:
            raise ValueError(
                f"Scenario {scenario_id} must provide one threat multiplier for every convoy."
            )
        if any(not _finite(v) or v <= 0 for v in scenario.threat_multipliers.values()):
            raise ValueError(
                f"Scenario {scenario_id} threat multipliers must be finite and positive."
            )

    if abs(probability_sum - 1.0) > _EPS:
        raise ValueError("Scenario probabilities must sum to 1.0.")

    max_slots = min(max_available_escorts, len(diminishing_returns), len(escorts))
    minimum_required = 0
    for convoy_id, convoy in convoys.items():
        if convoy.ships <= 0:
            raise ValueError(f"Convoy {convoy_id} must contain at least one ship.")
        if not _finite(convoy.threat) or convoy.threat <= 0:
            raise ValueError(f"Convoy {convoy_id} threat must be finite and positive.")
        if not _finite(convoy.baseline_survival) or not 0 <= convoy.baseline_survival <= 1:
            raise ValueError(f"Convoy {convoy_id} baseline_survival must be in [0, 1].")
        if convoy.min_escorts < 0:
            raise ValueError(f"Convoy {convoy_id} min_escorts must be non-negative.")
        upper = len(escorts) if convoy.max_escorts is None else convoy.max_escorts
        if convoy.min_escorts > upper:
            raise ValueError(f"Convoy {convoy_id} min_escorts cannot exceed max_escorts.")
        if convoy.min_escorts > max_slots:
            raise ValueError(
                f"Convoy {convoy_id} min_escorts exceeds the number of modeled escort slots."
            )
        minimum_required += convoy.min_escorts

    if minimum_required > max_available_escorts:
        raise ValueError(
            "The sum of convoy minimum escort requirements exceeds max_available_escorts."
        )

    for escort_id, escort in escorts.items():
        if not _finite(escort.protection) or escort.protection <= 0:
            raise ValueError(f"Escort {escort_id} protection must be finite and positive.")


def _validate_cvar_parameters(cvar_alpha: float, cvar_weight: float) -> None:
    if not _finite(cvar_alpha) or not 0 <= cvar_alpha < 1:
        raise ValueError("cvar_alpha must be in [0, 1).")
    if not _finite(cvar_weight) or not 0 <= cvar_weight <= 1:
        raise ValueError("cvar_weight must be in [0, 1].")


def _build_uncertain_core(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    scenarios: Mapping[str, ThreatScenario],
    max_available_escorts: int,
    diminishing_returns: Sequence[float],
    effectiveness_scale: float,
    model_name: str,
):
    convoy_ids = tuple(convoys)
    escort_ids = tuple(escorts)
    scenario_ids = tuple(scenarios)
    max_slots = min(max_available_escorts, len(diminishing_returns), len(escort_ids))

    model = LpProblem(model_name, LpMaximize)

    x = {
        (c, e): LpVariable(f"assign_{i}_{j}", cat=LpBinary)
        for i, c in enumerate(convoy_ids)
        for j, e in enumerate(escort_ids)
    }
    z = {
        (c, e, k): LpVariable(f"slot_{i}_{j}_{k}", cat=LpBinary)
        for i, c in enumerate(convoy_ids)
        for j, e in enumerate(escort_ids)
        for k in range(max_slots)
    }
    slot_used = {
        (c, k): LpVariable(f"slot_used_{i}_{k}", cat=LpBinary)
        for i, c in enumerate(convoy_ids)
        for k in range(max_slots)
    }

    gain: dict[tuple[str, str, str, int], float] = {}
    for s in scenario_ids:
        scenario = scenarios[s]
        for c in convoy_ids:
            convoy = convoys[c]
            scenario_threat = convoy.threat * scenario.threat_multipliers[c]
            remaining = 1.0 - convoy.baseline_survival
            for e in escort_ids:
                for k in range(max_slots):
                    raw = (
                        effectiveness_scale
                        * escorts[e].protection
                        * diminishing_returns[k]
                        / scenario_threat
                    )
                    gain[s, c, e, k] = min(raw, remaining)

    baseline = sum(convoys[c].ships * convoys[c].baseline_survival for c in convoy_ids)
    scenario_totals = {
        s: baseline
        + lpSum(
            convoys[c].ships * gain[s, c, e, k] * z[c, e, k]
            for c in convoy_ids
            for e in escort_ids
            for k in range(max_slots)
        )
        for s in scenario_ids
    }

    for e in escort_ids:
        model += lpSum(x[c, e] for c in convoy_ids) <= 1
    model += lpSum(x.values()) <= max_available_escorts

    for c in convoy_ids:
        convoy = convoys[c]
        upper = len(escort_ids) if convoy.max_escorts is None else convoy.max_escorts
        upper = min(upper, max_slots)
        assigned_count = lpSum(x[c, e] for e in escort_ids)
        model += assigned_count >= convoy.min_escorts
        model += assigned_count <= upper
        model += lpSum(slot_used[c, k] for k in range(max_slots)) == assigned_count

        for k in range(max_slots):
            model += lpSum(z[c, e, k] for e in escort_ids) == slot_used[c, k]
        for k in range(1, max_slots):
            model += slot_used[c, k] <= slot_used[c, k - 1]
        for e in escort_ids:
            model += lpSum(z[c, e, k] for k in range(max_slots)) == x[c, e]

        for s in scenario_ids:
            model += lpSum(
                gain[s, c, e, k] * z[c, e, k]
                for e in escort_ids
                for k in range(max_slots)
            ) <= 1.0 - convoy.baseline_survival

    return model, x, scenario_totals, convoy_ids, escort_ids, scenario_ids


def _extract_outcomes(
    convoys: Mapping[str, Convoy],
    scenarios: Mapping[str, ThreatScenario],
    scenario_totals,
    scenario_ids: Sequence[str],
) -> tuple[ScenarioOutcome, ...]:
    total_ships = sum(convoy.ships for convoy in convoys.values())
    outcomes = []
    for s in scenario_ids:
        realized = float(value(scenario_totals[s]))
        outcomes.append(
            ScenarioOutcome(
                scenario_id=s,
                probability=scenarios[s].probability,
                expected_survivors=realized,
                survival_rate=realized / total_ships,
            )
        )
    return tuple(outcomes)


def _lower_tail_cvar(outcomes: Sequence[ScenarioOutcome], alpha: float) -> float:
    """Return lower-tail CVaR for a discrete survivor distribution.

    The tail mass is 1 - alpha. Outcomes are accumulated from the smallest
    survivor count upward, allowing the final scenario probability to be used
    fractionally when it crosses the tail-mass boundary.
    """
    tail_mass = 1.0 - alpha
    remaining = tail_mass
    weighted_sum = 0.0

    for outcome in sorted(outcomes, key=lambda item: item.expected_survivors):
        if remaining <= _EPS:
            break
        mass = min(outcome.probability, remaining)
        weighted_sum += mass * outcome.expected_survivors
        remaining -= mass

    if remaining > 1e-7:
        raise RuntimeError("Scenario probabilities do not cover the requested CVaR tail.")
    return weighted_sum / tail_mass


def solve_uncertain_allocation(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    scenarios: Mapping[str, ThreatScenario],
    max_available_escorts: int,
    *,
    diminishing_returns: Sequence[float] = (1.0, 0.72, 0.52, 0.38, 0.28, 0.20),
    effectiveness_scale: float = 0.9,
    risk_aversion: float = 0.0,
) -> UncertainAllocationResult:
    """Solve a scenario-based allocation MILP under threat uncertainty.

    Assignment decisions are first-stage decisions shared by all scenarios.
    The objective is a convex combination of probability-weighted expected
    survivors and worst-case survivors.

    risk_aversion = 0.0 gives a risk-neutral stochastic program.
    risk_aversion = 1.0 gives a maximin robust allocation over the supplied
    finite scenario set.
    """
    _validate_uncertain_inputs(
        convoys,
        escorts,
        scenarios,
        max_available_escorts,
        diminishing_returns,
        effectiveness_scale,
        risk_aversion,
    )

    model, x, scenario_totals, convoy_ids, escort_ids, scenario_ids = _build_uncertain_core(
        convoys,
        escorts,
        scenarios,
        max_available_escorts,
        diminishing_returns,
        effectiveness_scale,
        "Uncertain_Convoy_Escort_Allocation",
    )

    worst_case = LpVariable("worst_case_expected_survivors", lowBound=0)
    expected_total = lpSum(
        scenarios[s].probability * scenario_totals[s] for s in scenario_ids
    )
    model += (1.0 - risk_aversion) * expected_total + risk_aversion * worst_case

    for s in scenario_ids:
        model += worst_case <= scenario_totals[s]

    model.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not return an optimal solution: {status}")

    assigned = {
        c: tuple(e for e in escort_ids if (x[c, e].value() or 0.0) > 0.5)
        for c in convoy_ids
    }
    outcomes = _extract_outcomes(convoys, scenarios, scenario_totals, scenario_ids)
    expected_survivors = sum(item.probability * item.expected_survivors for item in outcomes)
    worst_case_survivors = min(item.expected_survivors for item in outcomes)
    reported_objective = (
        (1.0 - risk_aversion) * expected_survivors
        + risk_aversion * worst_case_survivors
    )

    return UncertainAllocationResult(
        status=status,
        objective_value=reported_objective,
        risk_aversion=risk_aversion,
        assigned_escorts=assigned,
        scenarios=outcomes,
    )


def solve_cvar_allocation(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    scenarios: Mapping[str, ThreatScenario],
    max_available_escorts: int,
    *,
    diminishing_returns: Sequence[float] = (1.0, 0.72, 0.52, 0.38, 0.28, 0.20),
    effectiveness_scale: float = 0.9,
    cvar_alpha: float = 0.90,
    cvar_weight: float = 1.0,
) -> CVaRAllocationResult:
    """Solve a lower-tail CVaR-aware allocation MILP.

    CVaR is applied to the distribution of scenario survivor totals. Because
    larger survivor totals are preferred, this formulation maximizes the mean
    of the worst 1 - cvar_alpha probability mass. cvar_weight controls the
    trade-off between probability-weighted expected survivors and lower-tail
    CVaR. A weight of zero is risk-neutral; a weight of one is pure CVaR.
    """
    _validate_uncertain_inputs(
        convoys,
        escorts,
        scenarios,
        max_available_escorts,
        diminishing_returns,
        effectiveness_scale,
        risk_aversion=0.0,
    )
    _validate_cvar_parameters(cvar_alpha, cvar_weight)

    model, x, scenario_totals, convoy_ids, escort_ids, scenario_ids = _build_uncertain_core(
        convoys,
        escorts,
        scenarios,
        max_available_escorts,
        diminishing_returns,
        effectiveness_scale,
        "CVaR_Convoy_Escort_Allocation",
    )

    expected_total = lpSum(
        scenarios[s].probability * scenario_totals[s] for s in scenario_ids
    )

    eta = LpVariable("cvar_tail_threshold", lowBound=0)
    shortfall = {
        s: LpVariable(f"cvar_shortfall_{i}", lowBound=0)
        for i, s in enumerate(scenario_ids)
    }
    for s in scenario_ids:
        model += shortfall[s] >= eta - scenario_totals[s]

    cvar_expr = eta - (1.0 / (1.0 - cvar_alpha)) * lpSum(
        scenarios[s].probability * shortfall[s] for s in scenario_ids
    )
    model += (1.0 - cvar_weight) * expected_total + cvar_weight * cvar_expr

    model.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not return an optimal solution: {status}")

    assigned = {
        c: tuple(e for e in escort_ids if (x[c, e].value() or 0.0) > 0.5)
        for c in convoy_ids
    }
    outcomes = _extract_outcomes(convoys, scenarios, scenario_totals, scenario_ids)
    expected_survivors = sum(item.probability * item.expected_survivors for item in outcomes)
    cvar_survivors = _lower_tail_cvar(outcomes, cvar_alpha)
    reported_objective = (
        (1.0 - cvar_weight) * expected_survivors
        + cvar_weight * cvar_survivors
    )

    return CVaRAllocationResult(
        status=status,
        objective_value=reported_objective,
        cvar_alpha=cvar_alpha,
        cvar_weight=cvar_weight,
        assigned_escorts=assigned,
        scenarios=outcomes,
    )
