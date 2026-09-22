import pytest

from convoy_allocation.model import Convoy, Escort, solve_allocation
from convoy_allocation.scenario import (
    build_demo_scenario,
    build_uncertain_demo_scenarios,
)
from convoy_allocation.uncertainty import (
    ThreatScenario,
    solve_cvar_allocation,
    solve_uncertain_allocation,
)


def test_risk_neutral_objective_matches_expected_survivors() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    result = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=0.0,
    )

    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(result.expected_survivors, abs=1e-8)


def test_full_risk_aversion_matches_worst_case_survivors() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    result = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=1.0,
    )

    assert result.objective_value == pytest.approx(result.worst_case_survivors, abs=1e-8)


def test_single_nominal_scenario_matches_deterministic_model() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = {
        "Nominal": ThreatScenario(
            probability=1.0,
            threat_multipliers={convoy_id: 1.0 for convoy_id in convoys},
        )
    }

    deterministic = solve_allocation(convoys, escorts, limit)
    uncertain = solve_uncertain_allocation(convoys, escorts, scenarios, limit)

    assert uncertain.expected_survivors == pytest.approx(
        deterministic.expected_survivors,
        abs=1e-8,
    )


def test_uncertain_assignment_respects_escort_exclusivity() -> None:
    convoys, escorts, limit = build_demo_scenario()
    result = solve_uncertain_allocation(
        convoys,
        escorts,
        build_uncertain_demo_scenarios(),
        limit,
    )

    assigned = [
        escort_id
        for escort_ids in result.assigned_escorts.values()
        for escort_id in escort_ids
    ]
    assert len(assigned) == len(set(assigned))
    assert len(assigned) <= limit


def test_worst_case_solution_cannot_have_lower_worst_case_than_risk_neutral_solution() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    neutral = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=0.0,
    )
    robust = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=1.0,
    )

    assert robust.worst_case_survivors + 1e-8 >= neutral.worst_case_survivors


def test_cvar_alpha_zero_equals_expected_survivors() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    result = solve_cvar_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        cvar_alpha=0.0,
        cvar_weight=1.0,
    )

    assert result.status == "Optimal"
    assert result.cvar_survivors == pytest.approx(result.expected_survivors, abs=1e-8)
    assert result.objective_value == pytest.approx(result.expected_survivors, abs=1e-8)


def test_cvar_weight_zero_matches_risk_neutral_expected_value() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    neutral = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=0.0,
    )
    cvar_disabled = solve_cvar_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        cvar_alpha=0.90,
        cvar_weight=0.0,
    )

    assert cvar_disabled.expected_survivors == pytest.approx(
        neutral.expected_survivors,
        abs=1e-8,
    )
    assert cvar_disabled.objective_value == pytest.approx(
        neutral.objective_value,
        abs=1e-8,
    )


def test_cvar_is_between_worst_case_and_expected_value() -> None:
    convoys, escorts, limit = build_demo_scenario()
    result = solve_cvar_allocation(
        convoys,
        escorts,
        build_uncertain_demo_scenarios(),
        limit,
        cvar_alpha=0.75,
        cvar_weight=1.0,
    )

    assert result.worst_case_survivors <= result.cvar_survivors + 1e-8
    assert result.cvar_survivors <= result.expected_survivors + 1e-8


def test_higher_cvar_confidence_focuses_on_a_no_better_tail() -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()

    result = solve_cvar_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        cvar_alpha=0.90,
        cvar_weight=1.0,
    )

    cvar_50_for_same_solution = _discrete_lower_tail_cvar(result, 0.50)
    assert result.cvar_survivors <= cvar_50_for_same_solution + 1e-8


def _discrete_lower_tail_cvar(result, alpha: float) -> float:
    tail_mass = 1.0 - alpha
    remaining = tail_mass
    total = 0.0
    for outcome in sorted(result.scenarios, key=lambda item: item.expected_survivors):
        mass = min(remaining, outcome.probability)
        total += mass * outcome.expected_survivors
        remaining -= mass
        if remaining <= 1e-12:
            break
    return total / tail_mass


def test_rejects_scenario_probabilities_that_do_not_sum_to_one() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {
        "S1": ThreatScenario(0.6, {"Convoy-A": 1.0}),
        "S2": ThreatScenario(0.3, {"Convoy-A": 1.2}),
    }

    with pytest.raises(ValueError, match="sum to 1.0"):
        solve_uncertain_allocation(convoys, escorts, scenarios, 1)


def test_rejects_incomplete_scenario_multiplier_map() -> None:
    convoys = {
        "Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5),
        "Convoy-B": Convoy(ships=10, threat=5.0, baseline_survival=0.5),
    }
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {"S1": ThreatScenario(1.0, {"Convoy-A": 1.0})}

    with pytest.raises(ValueError, match="every convoy"):
        solve_uncertain_allocation(convoys, escorts, scenarios, 1)


def test_rejects_risk_aversion_outside_unit_interval() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {"S1": ThreatScenario(1.0, {"Convoy-A": 1.0})}

    with pytest.raises(ValueError, match="risk_aversion"):
        solve_uncertain_allocation(
            convoys,
            escorts,
            scenarios,
            1,
            risk_aversion=1.1,
        )


@pytest.mark.parametrize("alpha", [-0.01, 1.0, 1.2])
def test_rejects_invalid_cvar_alpha(alpha: float) -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {"S1": ThreatScenario(1.0, {"Convoy-A": 1.0})}

    with pytest.raises(ValueError, match="cvar_alpha"):
        solve_cvar_allocation(convoys, escorts, scenarios, 1, cvar_alpha=alpha)


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_rejects_invalid_cvar_weight(weight: float) -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}
    scenarios = {"S1": ThreatScenario(1.0, {"Convoy-A": 1.0})}

    with pytest.raises(ValueError, match="cvar_weight"):
        solve_cvar_allocation(convoys, escorts, scenarios, 1, cvar_weight=weight)
