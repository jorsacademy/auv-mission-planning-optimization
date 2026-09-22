import math

import pytest

from convoy_allocation.model import Convoy, Escort, solve_allocation
from convoy_allocation.scenario import build_demo_scenario


def test_demo_scenario_solves_optimally() -> None:
    convoys, escorts, max_available_escorts = build_demo_scenario()
    result = solve_allocation(convoys, escorts, max_available_escorts)

    assert result.status == "Optimal"
    assert result.total_ships == sum(item.ships for item in convoys.values())
    assert 0.0 <= result.survival_rate <= 1.0
    assert result.expected_survivors == pytest.approx(result.objective_value, abs=1e-8)


def test_each_escort_is_used_at_most_once() -> None:
    convoys, escorts, max_available_escorts = build_demo_scenario()
    result = solve_allocation(convoys, escorts, max_available_escorts)

    assigned = [escort for convoy in result.convoys for escort in convoy.assigned_escorts]
    assert len(assigned) == len(set(assigned))
    assert len(assigned) <= max_available_escorts


def test_survival_probability_respects_bounds() -> None:
    convoys, escorts, max_available_escorts = build_demo_scenario()
    result = solve_allocation(convoys, escorts, max_available_escorts)

    for convoy in result.convoys:
        assert convoy.baseline_survival <= convoy.survival_probability <= 1.0
        assert convoy.expected_losses >= 0.0
        assert convoy.expected_survivors + convoy.expected_losses == pytest.approx(
            convoy.ships
        )


def test_minimum_escort_requirement_is_enforced() -> None:
    convoys = {
        "Convoy-A": Convoy(
            ships=20,
            threat=5.0,
            baseline_survival=0.4,
            min_escorts=1,
            max_escorts=2,
        ),
        "Convoy-B": Convoy(
            ships=20,
            threat=5.0,
            baseline_survival=0.4,
            min_escorts=1,
            max_escorts=2,
        ),
    }
    escorts = {
        "Escort-01": Escort(protection=0.8),
        "Escort-02": Escort(protection=0.7),
    }

    result = solve_allocation(convoys, escorts, max_available_escorts=2)
    by_id = {item.convoy_id: item for item in result.convoys}

    assert len(by_id["Convoy-A"].assigned_escorts) >= 1
    assert len(by_id["Convoy-B"].assigned_escorts) >= 1


def test_maximum_escort_requirement_is_enforced() -> None:
    convoys = {
        "Convoy-A": Convoy(
            ships=100,
            threat=2.0,
            baseline_survival=0.2,
            max_escorts=1,
        ),
        "Convoy-B": Convoy(
            ships=10,
            threat=9.0,
            baseline_survival=0.5,
            max_escorts=2,
        ),
    }
    escorts = {
        "Escort-01": Escort(protection=0.8),
        "Escort-02": Escort(protection=0.7),
    }

    result = solve_allocation(convoys, escorts, max_available_escorts=2)
    by_id = {item.convoy_id: item for item in result.convoys}

    assert len(by_id["Convoy-A"].assigned_escorts) <= 1


def test_stronger_escort_is_preferred_in_single_slot_case() -> None:
    convoys = {
        "Convoy-A": Convoy(
            ships=50,
            threat=5.0,
            baseline_survival=0.4,
            max_escorts=1,
        )
    }
    escorts = {
        "Escort-Weak": Escort(protection=0.2),
        "Escort-Strong": Escort(protection=0.9),
    }

    result = solve_allocation(convoys, escorts, max_available_escorts=1)

    assert result.convoys[0].assigned_escorts == ("Escort-Strong",)


def test_diminishing_returns_reduce_second_slot_value() -> None:
    convoys = {
        "Convoy-A": Convoy(
            ships=40,
            threat=5.0,
            baseline_survival=0.4,
            max_escorts=2,
        )
    }
    escorts = {
        "Escort-01": Escort(protection=0.5),
        "Escort-02": Escort(protection=0.5),
    }

    one = solve_allocation(
        convoys,
        escorts,
        max_available_escorts=1,
        diminishing_returns=(1.0, 0.5),
        effectiveness_scale=1.0,
    )
    two = solve_allocation(
        convoys,
        escorts,
        max_available_escorts=2,
        diminishing_returns=(1.0, 0.5),
        effectiveness_scale=1.0,
    )

    first_increment = one.expected_survivors - 40 * 0.4
    second_increment = two.expected_survivors - one.expected_survivors
    assert second_increment < first_increment


def test_rejects_invalid_resource_limit() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}

    with pytest.raises(ValueError, match="cannot exceed"):
        solve_allocation(convoys, escorts, max_available_escorts=2)


def test_rejects_aggregate_minimum_above_resource_limit() -> None:
    convoys = {
        "Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5, min_escorts=1),
        "Convoy-B": Convoy(ships=10, threat=5.0, baseline_survival=0.5, min_escorts=1),
    }
    escorts = {
        "Escort-01": Escort(protection=0.5),
        "Escort-02": Escort(protection=0.5),
    }

    with pytest.raises(ValueError, match="sum of convoy minimum"):
        solve_allocation(convoys, escorts, max_available_escorts=1)


def test_rejects_minimum_above_modeled_slots() -> None:
    convoys = {
        "Convoy-A": Convoy(
            ships=10,
            threat=5.0,
            baseline_survival=0.5,
            min_escorts=2,
        )
    }
    escorts = {
        "Escort-01": Escort(protection=0.5),
        "Escort-02": Escort(protection=0.5),
    }

    with pytest.raises(ValueError, match="modeled escort slots"):
        solve_allocation(
            convoys,
            escorts,
            max_available_escorts=2,
            diminishing_returns=(1.0,),
        )


def test_rejects_increasing_return_schedule() -> None:
    convoys = {"Convoy-A": Convoy(ships=10, threat=5.0, baseline_survival=0.5)}
    escorts = {"Escort-01": Escort(protection=0.5)}

    with pytest.raises(ValueError, match="non-increasing"):
        solve_allocation(
            convoys,
            escorts,
            max_available_escorts=1,
            diminishing_returns=(0.5, 0.8),
        )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_numeric_inputs(bad_value: float) -> None:
    convoys = {
        "Convoy-A": Convoy(ships=10, threat=bad_value, baseline_survival=0.5)
    }
    escorts = {"Escort-01": Escort(protection=0.5)}

    with pytest.raises(ValueError, match="finite and positive"):
        solve_allocation(convoys, escorts, max_available_escorts=1)


def test_zero_resource_limit_returns_baseline_solution() -> None:
    convoys = {
        "Convoy-A": Convoy(ships=20, threat=5.0, baseline_survival=0.4),
        "Convoy-B": Convoy(ships=30, threat=7.0, baseline_survival=0.5),
    }
    escorts = {"Escort-01": Escort(protection=0.8)}

    result = solve_allocation(convoys, escorts, max_available_escorts=0)

    assert result.expected_survivors == pytest.approx(20 * 0.4 + 30 * 0.5)
    assert all(not item.assigned_escorts for item in result.convoys)
