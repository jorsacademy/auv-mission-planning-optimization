from __future__ import annotations

from .model import solve_allocation
from .scenario import build_demo_scenario


def main() -> None:
    convoys, escorts, max_available_escorts = build_demo_scenario()
    result = solve_allocation(convoys, escorts, max_available_escorts)

    print("RISK-AWARE CONVOY ESCORT ALLOCATION")
    print("=" * 72)
    print(f"Solver status: {result.status}")
    print(f"Objective value: {result.objective_value:.2f} expected survivors")
    print()

    for item in result.convoys:
        assigned = ", ".join(item.assigned_escorts) if item.assigned_escorts else "None"
        print(f"{item.convoy_id}: {item.ships} ships")
        print(f"  Assigned escorts: {assigned}")
        print(f"  Baseline survival probability: {item.baseline_survival:.3f}")
        print(f"  Optimized survival probability: {item.survival_probability:.3f}")
        print(f"  Expected survivors: {item.expected_survivors:.2f}")
        print(f"  Expected losses: {item.expected_losses:.2f}")
        print("-" * 72)

    print(f"Total ships: {result.total_ships}")
    print(f"Expected survivors: {result.expected_survivors:.2f}")
    print(f"Expected losses: {result.expected_losses:.2f}")
    print(f"Expected survival rate: {100 * result.survival_rate:.2f}%")


if __name__ == "__main__":
    main()
