from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random

from .data import Instance


@dataclass
class HeuristicSolution:
    status: str
    objective: float
    routes: Dict[str, list[str]]
    warehouses: Dict[str, str]
    iterations: int


def _route_metrics(
    instance: Instance,
    drone_name: str,
    depot: str,
    customers: List[str],
) -> tuple[float, float, float, bool]:
    drone = next(d for d in instance.drones if d.name == drone_name)
    customer_map = {c.name: c for c in instance.customers}

    route = [depot] + customers + [depot]
    distance = sum(
        instance.distance[route[i], route[i + 1]]
        for i in range(len(route) - 1)
    )
    payload = sum(customer_map[c].demand for c in customers)

    current_time = 0.0
    feasible = payload <= drone.capacity + 1e-9

    for i, customer_name in enumerate(customers):
        previous = depot if i == 0 else customers[i - 1]
        current_time += instance.travel_time[previous, customer_name]
        customer = customer_map[customer_name]

        if current_time < customer.earliest:
            current_time = customer.earliest
        if current_time > customer.latest + 1e-9:
            feasible = False

        current_time += customer.service_time

    if customers:
        current_time += instance.travel_time[customers[-1], depot]

    feasible = feasible and current_time <= drone.max_route_time + 1e-9

    deployment_cost = next(
        w.deployment_cost for w in instance.warehouses if w.name == depot
    )
    cost = distance * drone.cost_per_distance + deployment_cost
    return cost, distance, current_time, feasible


def _total_cost(
    instance: Instance,
    routes: Dict[str, list[str]],
    warehouses: Dict[str, str],
) -> tuple[float, bool]:
    total = 0.0

    for drone_name, customers in routes.items():
        if not customers:
            continue
        cost, _, _, feasible = _route_metrics(
            instance,
            drone_name,
            warehouses[drone_name],
            customers,
        )
        if not feasible:
            return float("inf"), False
        total += cost

    served = sorted(c for route in routes.values() for c in route)
    return total, served == sorted(instance.customer_names)


def _best_initial_insertion(
    instance: Instance,
    rng: random.Random,
) -> tuple[Dict[str, list[str]], Dict[str, str]]:
    routes = {d.name: [] for d in instance.drones}

    warehouses = {
        d.name: min(
            instance.warehouses,
            key=lambda w: w.deployment_cost
            + sum(
                instance.distance[w.name, c.name]
                for c in instance.customers
            )
            / len(instance.customers),
        ).name
        for d in instance.drones
    }

    customers = list(instance.customer_names)
    rng.shuffle(customers)
    customers.sort(
        key=lambda name: next(
            c.demand for c in instance.customers if c.name == name
        ),
        reverse=True,
    )

    for customer_name in customers:
        best = None

        for drone_name in instance.drone_names:
            for warehouse_name in instance.warehouse_names:
                for position in range(len(routes[drone_name]) + 1):
                    candidate = (
                        routes[drone_name][:position]
                        + [customer_name]
                        + routes[drone_name][position:]
                    )

                    candidate_cost, _, _, feasible = _route_metrics(
                        instance,
                        drone_name,
                        warehouse_name,
                        candidate,
                    )
                    if not feasible:
                        continue

                    other_cost = 0.0
                    for other_drone, other_customers in routes.items():
                        if other_drone == drone_name or not other_customers:
                            continue
                        cost, _, _, ok = _route_metrics(
                            instance,
                            other_drone,
                            warehouses[other_drone],
                            other_customers,
                        )
                        if not ok:
                            other_cost = float("inf")
                            break
                        other_cost += cost

                    score = candidate_cost + other_cost
                    if best is None or score < best[0]:
                        best = (
                            score,
                            drone_name,
                            warehouse_name,
                            position,
                        )

        if best is None:
            raise RuntimeError(
                f"Could not construct a feasible initial solution for {customer_name}."
            )

        _, drone_name, warehouse_name, position = best
        warehouses[drone_name] = warehouse_name
        routes[drone_name].insert(position, customer_name)

    return routes, warehouses


def solve_hill_climbing(
    instance: Instance,
    seed: int = 7,
    max_iterations: int = 500,
) -> HeuristicSolution:
    rng = random.Random(seed)
    routes, warehouses = _best_initial_insertion(instance, rng)
    best_cost, feasible = _total_cost(instance, routes, warehouses)

    if not feasible:
        raise RuntimeError("Initial heuristic solution is infeasible.")

    iterations = 0
    improved = True

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        # Neighborhood 1: 2-opt reversal inside one drone route.
        for drone_name in instance.drone_names:
            route = routes[drone_name]
            for i in range(len(route)):
                for j in range(i + 2, len(route) + 1):
                    candidate_routes = {
                        d: customers[:] for d, customers in routes.items()
                    }
                    candidate_routes[drone_name] = (
                        route[:i]
                        + list(reversed(route[i:j]))
                        + route[j:]
                    )

                    cost, ok = _total_cost(
                        instance,
                        candidate_routes,
                        warehouses,
                    )
                    if ok and cost + 1e-9 < best_cost:
                        routes = candidate_routes
                        best_cost = cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Neighborhood 2: relocate one customer between drone routes.
        for source_drone in instance.drone_names:
            for index, customer_name in enumerate(routes[source_drone]):
                for target_drone in instance.drone_names:
                    if source_drone == target_drone:
                        continue

                    for position in range(len(routes[target_drone]) + 1):
                        candidate_routes = {
                            d: customers[:] for d, customers in routes.items()
                        }
                        candidate_routes[source_drone].pop(index)
                        candidate_routes[target_drone].insert(
                            position,
                            customer_name,
                        )

                        for warehouse_name in instance.warehouse_names:
                            candidate_warehouses = warehouses.copy()
                            candidate_warehouses[target_drone] = warehouse_name

                            cost, ok = _total_cost(
                                instance,
                                candidate_routes,
                                candidate_warehouses,
                            )
                            if ok and cost + 1e-9 < best_cost:
                                routes = {
                                    d: customers[:]
                                    for d, customers in candidate_routes.items()
                                }
                                warehouses = candidate_warehouses
                                best_cost = cost
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    routes = {d: route for d, route in routes.items() if route}
    warehouses = {d: warehouses[d] for d in routes}

    return HeuristicSolution(
        status="Feasible",
        objective=best_cost,
        routes=routes,
        warehouses=warehouses,
        iterations=iterations,
    )
