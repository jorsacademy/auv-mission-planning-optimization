from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from .data import Instance


def plot_routes(
    instance: Instance,
    routes: dict[str, list[str]],
    warehouses: dict[str, str],
    output_path: str | Path,
) -> None:
    coords = {w.name: (w.x, w.y) for w in instance.warehouses}
    coords.update({c.name: (c.x, c.y) for c in instance.customers})

    fig, ax = plt.subplots(figsize=(9, 7))

    for warehouse in instance.warehouses:
        ax.scatter(warehouse.x, warehouse.y, marker="s", s=120)
        ax.text(warehouse.x + 0.15, warehouse.y + 0.15, warehouse.name)

    for customer in instance.customers:
        ax.scatter(customer.x, customer.y, marker="o", s=60)
        ax.text(customer.x + 0.15, customer.y + 0.15, customer.name)

    for drone_name, route in routes.items():
        depot = warehouses[drone_name]
        full_route = route if route and route[0] == depot else [depot] + route + [depot]
        if full_route[-1] != depot:
            full_route.append(depot)

        xs = [coords[node][0] for node in full_route]
        ys = [coords[node][1] for node in full_route]
        ax.plot(xs, ys, marker="o", label=drone_name)

    ax.set_title("Drone Delivery Routes")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
