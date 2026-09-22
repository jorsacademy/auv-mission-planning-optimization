from __future__ import annotations

from .scenario import build_demo_scenario, build_uncertain_demo_scenarios
from .uncertainty import solve_cvar_allocation, solve_uncertain_allocation


def _print_result(label: str, risk_aversion: float) -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()
    result = solve_uncertain_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        risk_aversion=risk_aversion,
    )

    print(label)
    print("=" * 72)
    print(f"Solver status: {result.status}")
    print(f"Risk aversion: {result.risk_aversion:.2f}")
    print(f"Objective value: {result.objective_value:.2f}")
    print(f"Probability-weighted expected survivors: {result.expected_survivors:.2f}")
    print(f"Worst-case expected survivors: {result.worst_case_survivors:.2f}")
    print()

    for convoy_id, escort_ids in result.assigned_escorts.items():
        assigned = ", ".join(escort_ids) if escort_ids else "None"
        print(f"{convoy_id}: {assigned}")

    print()
    print("Scenario outcomes")
    print("-" * 72)
    for outcome in result.scenarios:
        print(
            f"{outcome.scenario_id}: probability={outcome.probability:.2f}, "
            f"expected_survivors={outcome.expected_survivors:.2f}, "
            f"survival_rate={100 * outcome.survival_rate:.2f}%"
        )
    print()


def _print_cvar_result(cvar_alpha: float, cvar_weight: float) -> None:
    convoys, escorts, limit = build_demo_scenario()
    scenarios = build_uncertain_demo_scenarios()
    result = solve_cvar_allocation(
        convoys,
        escorts,
        scenarios,
        limit,
        cvar_alpha=cvar_alpha,
        cvar_weight=cvar_weight,
    )

    print("LOWER-TAIL CVAR ALLOCATION")
    print("=" * 72)
    print(f"Solver status: {result.status}")
    print(f"CVaR confidence level: {result.cvar_alpha:.2f}")
    print(f"CVaR objective weight: {result.cvar_weight:.2f}")
    print(f"Objective value: {result.objective_value:.2f}")
    print(f"Probability-weighted expected survivors: {result.expected_survivors:.2f}")
    print(f"Lower-tail CVaR survivors: {result.cvar_survivors:.2f}")
    print(f"Worst-case expected survivors: {result.worst_case_survivors:.2f}")
    print()

    for convoy_id, escort_ids in result.assigned_escorts.items():
        assigned = ", ".join(escort_ids) if escort_ids else "None"
        print(f"{convoy_id}: {assigned}")

    print()
    print("Scenario outcomes")
    print("-" * 72)
    for outcome in result.scenarios:
        print(
            f"{outcome.scenario_id}: probability={outcome.probability:.2f}, "
            f"expected_survivors={outcome.expected_survivors:.2f}, "
            f"survival_rate={100 * outcome.survival_rate:.2f}%"
        )
    print()


def main() -> None:
    _print_result("RISK-NEUTRAL STOCHASTIC ALLOCATION", risk_aversion=0.0)
    _print_result("ROBUST MAXIMIN ALLOCATION", risk_aversion=1.0)
    _print_cvar_result(cvar_alpha=0.90, cvar_weight=0.75)


if __name__ == "__main__":
    main()
