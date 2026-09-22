from pathlib import Path

from drone_delivery import load_instance, solve_milp
from drone_delivery.reporting import print_solution
from drone_delivery.visualization import plot_routes


def main() -> None:
    root = Path(__file__).resolve().parent
    instance = load_instance(root / "data")
    solution = solve_milp(instance, msg=False, time_limit=60)

    print(f"Solver status: {solution.status}")
    if solution.objective is None:
        return

    print_solution(
        "MILP Solution",
        instance,
        solution.objective,
        solution.routes,
        solution.warehouses,
    )
    plot_routes(
        instance,
        solution.routes,
        solution.warehouses,
        root / "milp_routes.png",
    )
    print("Route plot saved to milp_routes.png")


if __name__ == "__main__":
    main()
