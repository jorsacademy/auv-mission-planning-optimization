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
)


@dataclass(frozen=True)
class Convoy:
    """Synthetic convoy input data."""

    ships: int
    threat: float
    baseline_survival: float
    min_escorts: int = 0
    max_escorts: int | None = None


@dataclass(frozen=True)
class Escort:
    """Synthetic escort input data."""

    protection: float


@dataclass(frozen=True)
class ConvoyResult:
    convoy_id: str
    ships: int
    assigned_escorts: tuple[str, ...]
    baseline_survival: float
    survival_probability: float
    expected_survivors: float
    expected_losses: float


@dataclass(frozen=True)
class AllocationResult:
    status: str
    objective_value: float
    convoys: tuple[ConvoyResult, ...]

    @property
    def total_ships(self) -> int:
        return sum(item.ships for item in self.convoys)

    @property
    def expected_survivors(self) -> float:
        return sum(item.expected_survivors for item in self.convoys)

    @property
    def expected_losses(self) -> float:
        return sum(item.expected_losses for item in self.convoys)

    @property
    def survival_rate(self) -> float:
        if self.total_ships == 0:
            return 0.0
        return self.expected_survivors / self.total_ships


def _is_finite(value_: float) -> bool:
    return math.isfinite(float(value_))


def _validate_inputs(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    max_available_escorts: int,
    diminishing_returns: Sequence[float],
) -> None:
    if not convoys:
        raise ValueError("At least one convoy is required.")
    if not escorts:
        raise ValueError("At least one escort is required.")
    if not isinstance(max_available_escorts, int) or isinstance(max_available_escorts, bool):
        raise TypeError("max_available_escorts must be an integer.")
    if max_available_escorts < 0:
        raise ValueError("max_available_escorts must be non-negative.")
    if max_available_escorts > len(escorts):
        raise ValueError("max_available_escorts cannot exceed the number of escorts.")
    if not diminishing_returns:
        raise ValueError("diminishing_returns must contain at least one value.")
    if any(not _is_finite(value_) for value_ in diminishing_returns):
        raise ValueError("diminishing_returns values must be finite.")
    if any(value_ <= 0 for value_ in diminishing_returns):
        raise ValueError("diminishing_returns values must be positive.")
    if any(a < b for a, b in zip(diminishing_returns, diminishing_returns[1:])):
        raise ValueError("diminishing_returns must be non-increasing.")

    max_slots = min(max_available_escorts, len(diminishing_returns), len(escorts))
    minimum_required = 0

    for convoy_id, convoy in convoys.items():
        if not isinstance(convoy_id, str) or not convoy_id:
            raise ValueError("Convoy identifiers must be non-empty strings.")
        if not isinstance(convoy.ships, int) or isinstance(convoy.ships, bool):
            raise TypeError(f"Convoy {convoy_id} ships must be an integer.")
        if convoy.ships <= 0:
            raise ValueError(f"Convoy {convoy_id} must contain at least one ship.")
        if not _is_finite(convoy.threat) or convoy.threat <= 0:
            raise ValueError(f"Convoy {convoy_id} threat must be finite and positive.")
        if not _is_finite(convoy.baseline_survival):
            raise ValueError(f"Convoy {convoy_id} baseline_survival must be finite.")
        if not 0 <= convoy.baseline_survival <= 1:
            raise ValueError(f"Convoy {convoy_id} baseline_survival must be in [0, 1].")
        if not isinstance(convoy.min_escorts, int) or isinstance(convoy.min_escorts, bool):
            raise TypeError(f"Convoy {convoy_id} min_escorts must be an integer.")
        if convoy.min_escorts < 0:
            raise ValueError(f"Convoy {convoy_id} min_escorts must be non-negative.")

        upper = len(escorts) if convoy.max_escorts is None else convoy.max_escorts
        if not isinstance(upper, int) or isinstance(upper, bool):
            raise TypeError(f"Convoy {convoy_id} max_escorts must be an integer or None.")
        if upper < 0:
            raise ValueError(f"Convoy {convoy_id} max_escorts must be non-negative.")
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
        if not isinstance(escort_id, str) or not escort_id:
            raise ValueError("Escort identifiers must be non-empty strings.")
        if not _is_finite(escort.protection) or escort.protection <= 0:
            raise ValueError(f"Escort {escort_id} protection must be finite and positive.")


def _marginal_gain(
    convoy: Convoy,
    escort_protection: float,
    return_factor: float,
    effectiveness_scale: float,
) -> float:
    return effectiveness_scale * escort_protection * return_factor / convoy.threat


def _gain_coefficient(
    convoy: Convoy,
    escort: Escort,
    return_factor: float,
    effectiveness_scale: float,
) -> float:
    """Return the linear survival-probability coefficient for one escort-slot pair."""
    remaining_probability = 1.0 - convoy.baseline_survival
    raw_gain = _marginal_gain(
        convoy,
        escort.protection,
        return_factor,
        effectiveness_scale,
    )
    return min(raw_gain, remaining_probability)


def solve_allocation(
    convoys: Mapping[str, Convoy],
    escorts: Mapping[str, Escort],
    max_available_escorts: int,
    *,
    diminishing_returns: Sequence[float] = (1.0, 0.72, 0.52, 0.38, 0.28, 0.20),
    effectiveness_scale: float = 0.9,
) -> AllocationResult:
    """Solve the risk-aware escort allocation MILP.

    Each escort can be assigned to at most one convoy. Within a convoy, assigned
    escorts occupy ordered marginal-effect slots. Earlier slots have larger
    return factors, which creates diminishing returns while preserving a linear
    mixed-integer formulation.

    The exact same linear survival coefficients are used in the objective,
    feasibility constraints, and reported results.
    """
    _validate_inputs(convoys, escorts, max_available_escorts, diminishing_returns)
    if not _is_finite(effectiveness_scale) or effectiveness_scale <= 0:
        raise ValueError("effectiveness_scale must be finite and positive.")

    convoy_ids = tuple(convoys)
    escort_ids = tuple(escorts)
    convoy_index = {convoy_id: index for index, convoy_id in enumerate(convoy_ids)}
    escort_index = {escort_id: index for index, escort_id in enumerate(escort_ids)}
    max_slots = min(max_available_escorts, len(diminishing_returns), len(escort_ids))

    model = LpProblem("Risk_Aware_Convoy_Escort_Allocation", LpMaximize)

    x = {
        (convoy_id, escort_id): LpVariable(
            f"assign_c{convoy_index[convoy_id]}_e{escort_index[escort_id]}",
            cat=LpBinary,
        )
        for convoy_id in convoy_ids
        for escort_id in escort_ids
    }
    z = {
        (convoy_id, escort_id, slot): LpVariable(
            f"slot_c{convoy_index[convoy_id]}_e{escort_index[escort_id]}_k{slot}",
            cat=LpBinary,
        )
        for convoy_id in convoy_ids
        for escort_id in escort_ids
        for slot in range(max_slots)
    }
    slot_used = {
        (convoy_id, slot): LpVariable(
            f"slot_used_c{convoy_index[convoy_id]}_k{slot}", cat=LpBinary
        )
        for convoy_id in convoy_ids
        for slot in range(max_slots)
    }

    baseline_expected_survivors = sum(
        convoys[convoy_id].ships * convoys[convoy_id].baseline_survival
        for convoy_id in convoy_ids
    )

    gain = {
        (convoy_id, escort_id, slot): _gain_coefficient(
            convoys[convoy_id],
            escorts[escort_id],
            diminishing_returns[slot],
            effectiveness_scale,
        )
        for convoy_id in convoy_ids
        for escort_id in escort_ids
        for slot in range(max_slots)
    }

    model += baseline_expected_survivors + lpSum(
        convoys[convoy_id].ships
        * gain[convoy_id, escort_id, slot]
        * z[convoy_id, escort_id, slot]
        for convoy_id in convoy_ids
        for escort_id in escort_ids
        for slot in range(max_slots)
    )

    for escort_id in escort_ids:
        model += lpSum(x[convoy_id, escort_id] for convoy_id in convoy_ids) <= 1

    model += lpSum(x.values()) <= max_available_escorts

    for convoy_id in convoy_ids:
        convoy = convoys[convoy_id]
        max_for_convoy = len(escort_ids) if convoy.max_escorts is None else convoy.max_escorts
        max_for_convoy = min(max_for_convoy, max_slots)
        assigned_count = lpSum(x[convoy_id, escort_id] for escort_id in escort_ids)

        model += assigned_count >= convoy.min_escorts
        model += assigned_count <= max_for_convoy
        model += (
            lpSum(slot_used[convoy_id, slot] for slot in range(max_slots))
            == assigned_count
        )

        for slot in range(max_slots):
            model += (
                lpSum(z[convoy_id, escort_id, slot] for escort_id in escort_ids)
                == slot_used[convoy_id, slot]
            )

        for slot in range(1, max_slots):
            model += slot_used[convoy_id, slot] <= slot_used[convoy_id, slot - 1]

        for escort_id in escort_ids:
            model += (
                lpSum(z[convoy_id, escort_id, slot] for slot in range(max_slots))
                == x[convoy_id, escort_id]
            )

        model += (
            lpSum(
                gain[convoy_id, escort_id, slot]
                * z[convoy_id, escort_id, slot]
                for escort_id in escort_ids
                for slot in range(max_slots)
            )
            <= 1.0 - convoy.baseline_survival
        )

    model.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[model.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not return an optimal solution: {status}")

    convoy_results = []
    for convoy_id in convoy_ids:
        convoy = convoys[convoy_id]
        assigned = tuple(
            escort_id
            for escort_id in escort_ids
            if (x[convoy_id, escort_id].value() or 0.0) > 0.5
        )

        optimized_gain = sum(
            gain[convoy_id, escort_id, slot]
            for escort_id in escort_ids
            for slot in range(max_slots)
            if (z[convoy_id, escort_id, slot].value() or 0.0) > 0.5
        )
        survival_probability = convoy.baseline_survival + optimized_gain
        survival_probability = min(1.0, max(0.0, survival_probability))
        expected_survivors = convoy.ships * survival_probability
        expected_losses = convoy.ships - expected_survivors

        convoy_results.append(
            ConvoyResult(
                convoy_id=convoy_id,
                ships=convoy.ships,
                assigned_escorts=assigned,
                baseline_survival=convoy.baseline_survival,
                survival_probability=survival_probability,
                expected_survivors=expected_survivors,
                expected_losses=expected_losses,
            )
        )

    convoy_results_tuple = tuple(convoy_results)
    objective_value = sum(item.expected_survivors for item in convoy_results_tuple)
    return AllocationResult(
        status=status,
        objective_value=objective_value,
        convoys=convoy_results_tuple,
    )
