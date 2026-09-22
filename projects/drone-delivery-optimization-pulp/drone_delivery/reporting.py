from __future__ import annotations

from .data import Instance


def print_solution(
    title: str,
    instance: Instance,
    objective: float | None,
    routes: dict[str, list[str]],
    warehouses: dict[str, str],
) -> None:
    customer = {c.name: c for c in instance.customers}
    drones = {d.name: d for d in instance.drones}

    print(f"\n{title}")
    print("=" * len(title))
    if objective is not None:
        print(f"Objective: {objective:.3f}")

    for drone_name, route in routes.items():
        depot = warehouses[drone_name]
        full_route = route if route and route[0] == depot else [depot] + route + [depot]
        if full_route[-1] != depot:
            full_route.append(depot)

        distance = sum(
            instance.distance[full_route[i], full_route[i + 1]]
            for i in range(len(full_route) - 1)
        )
        payload = sum(customer[n].demand for n in full_route if n in customer)

        print(f"{drone_name}: {' -> '.join(full_route)}")
        print(
            f"  load={payload:.1f}/{drones[drone_name].capacity:.1f}, "
            f"distance={distance:.3f}"
        )
