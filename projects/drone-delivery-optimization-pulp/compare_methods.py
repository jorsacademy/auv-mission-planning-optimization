from pathlib import Path

from drone_delivery import load_instance, solve_hill_climbing, solve_milp
from drone_delivery.reporting import print_solution
from drone_delivery.visualization import plot_routes


def main() -> None:
    root = Path(__file__).resolve().parent
    instance = load_instance(root / "data")

    milp = solve_milp(instance, msg=False, time_limit=60)
    hill_climbing = solve_hill_climbing(
        instance,
        seed=7,
        max_iterations=500,
    )

    print_solution(
        "MILP Solution",
        instance,
        milp.objective,
        milp.routes,
        milp.warehouses,
    )
    print_solution(
        "Hill Climbing Baseline",
        instance,
        hill_climbing.objective,
        hill_climbing.routes,
        hill_climbing.warehouses,
    )

    if milp.objective is not None:
        gap = 100.0 * (
            hill_climbing.objective - milp.objective
        ) / milp.objective
        print(f"\nHeuristic gap relative to MILP objective: {gap:.2f}%")

    print(f"Hill Climbing iterations: {hill_climbing.iterations}")

    if milp.objective is not None:
        plot_routes(
            instance,
            milp.routes,
            milp.warehouses,
            root / "milp_routes.png",
        )
    plot_routes(
        instance,
        hill_climbing.routes,
        hill_climbing.warehouses,
        root / "hill_climbing_routes.png",
    )
    print("Route plots generated.")


if __name__ == "__main__":
    main()
