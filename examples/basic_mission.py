import matplotlib.pyplot as plt

from auv_optimizer import AUVOptimizer


def main() -> None:
    optimizer = AUVOptimizer(num_auvs=4, num_tasks=12, seed=42)

    try:
        solution = optimizer.solve(msg=True)
    except ValueError as exc:
        print(f"Generated instance is incompatible: {exc}")
        return

    if solution is None:
        print("No optimal solution found.")
        return

    optimizer.print_solution_summary()
    optimizer.visualize_solution()
    plt.show()


if __name__ == "__main__":
    main()
