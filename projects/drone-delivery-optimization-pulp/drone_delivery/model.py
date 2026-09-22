from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

from .data import Instance


@dataclass
class MilpSolution:
    status: str
    objective: float | None
    routes: Dict[str, list[str]]
    warehouses: Dict[str, str]
    route_costs: Dict[str, float]
    route_times: Dict[str, float]
    route_loads: Dict[str, float]


def build_model(instance: Instance) -> tuple[LpProblem, dict]:
    W = instance.warehouse_names
    C = instance.customer_names
    D = instance.drone_names
    N = instance.nodes

    customer = {c.name: c for c in instance.customers}
    drone = {d.name: d for d in instance.drones}
    warehouse = {w.name: w for w in instance.warehouses}
    arcs = [(i, j) for i in N for j in N if i != j]

    model = LpProblem("Multi_Depot_Drone_CVRPTW", LpMinimize)

    active = LpVariable.dicts("active", D, cat=LpBinary)
    deploy = LpVariable.dicts("deploy", [(w, d) for w in W for d in D], cat=LpBinary)
    assign = LpVariable.dicts("assign", [(d, c) for d in D for c in C], cat=LpBinary)
    x = LpVariable.dicts("arc", [(d, i, j) for d in D for i, j in arcs], cat=LpBinary)
    arrival = LpVariable.dicts("arrival", [(d, c) for d in D for c in C], lowBound=0)
    load = LpVariable.dicts("load", [(d, c) for d in D for c in C], lowBound=0)

    model += (
        lpSum(
            drone[d].cost_per_distance * instance.distance[i, j] * x[d, i, j]
            for d in D
            for i, j in arcs
        )
        + lpSum(warehouse[w].deployment_cost * deploy[w, d] for w in W for d in D)
    )

    for c in C:
        model += lpSum(assign[d, c] for d in D) == 1, f"serve_once_{c}"

    for d in D:
        model += lpSum(deploy[w, d] for w in W) == active[d], f"one_depot_{d}"

    for w in W:
        for d in D:
            model += lpSum(x[d, w, j] for j in N if j != w) == deploy[w, d], f"depart_{w}_{d}"
            model += lpSum(x[d, i, w] for i in N if i != w) == deploy[w, d], f"return_{w}_{d}"

    for d in D:
        for c in C:
            model += lpSum(x[d, c, j] for j in N if j != c) == assign[d, c], f"flow_out_{d}_{c}"
            model += lpSum(x[d, i, c] for i in N if i != c) == assign[d, c], f"flow_in_{d}_{c}"

    for d in D:
        model += (
            lpSum(customer[c].demand * assign[d, c] for c in C)
            <= drone[d].capacity * active[d]
        ), f"capacity_{d}"

    max_latest = max(c.latest for c in instance.customers)
    max_travel = max(instance.travel_time.values())
    max_service = max(c.service_time for c in instance.customers)
    big_m_time = max_latest + max_travel + max_service + max(d.max_route_time for d in instance.drones)

    for d in D:
        for c in C:
            model += arrival[d, c] >= customer[c].earliest * assign[d, c], f"tw_lb_{d}_{c}"
            model += arrival[d, c] <= customer[c].latest + big_m_time * (1 - assign[d, c]), f"tw_ub_{d}_{c}"

    for d in D:
        for c1 in C:
            for c2 in C:
                if c1 == c2:
                    continue
                model += (
                    arrival[d, c2]
                    >= arrival[d, c1]
                    + customer[c1].service_time
                    + instance.travel_time[c1, c2]
                    - big_m_time * (1 - x[d, c1, c2])
                ), f"time_prop_{d}_{c1}_{c2}"

    for d in D:
        for w in W:
            for c in C:
                model += (
                    arrival[d, c]
                    >= instance.travel_time[w, c] - big_m_time * (1 - x[d, w, c])
                ), f"depot_time_{d}_{w}_{c}"
                model += (
                    arrival[d, c]
                    + customer[c].service_time
                    + instance.travel_time[c, w]
                    <= drone[d].max_route_time + big_m_time * (1 - x[d, c, w])
                ), f"return_deadline_{d}_{c}_{w}"

    for d in D:
        model += (
            lpSum(instance.travel_time[i, j] * x[d, i, j] for i, j in arcs)
            + lpSum(customer[c].service_time * assign[d, c] for c in C)
            <= drone[d].max_route_time * active[d]
        ), f"route_time_{d}"

    for d in D:
        Q = drone[d].capacity
        for c in C:
            q = customer[c].demand
            model += load[d, c] >= q * assign[d, c], f"load_lb_{d}_{c}"
            model += load[d, c] <= Q * assign[d, c], f"load_ub_{d}_{c}"

        for c1 in C:
            for c2 in C:
                if c1 == c2:
                    continue
                model += (
                    load[d, c2]
                    >= load[d, c1] + customer[c2].demand - Q * (1 - x[d, c1, c2])
                ), f"mtz_load_{d}_{c1}_{c2}"

        for w in W:
            for c in C:
                model += (
                    load[d, c] >= customer[c].demand - Q * (1 - x[d, w, c])
                ), f"depot_load_{d}_{w}_{c}"

    return model, {
        "active": active,
        "deploy": deploy,
        "assign": assign,
        "x": x,
        "arrival": arrival,
        "load": load,
    }


def _reconstruct_route(instance: Instance, drone_name: str, variables: dict) -> tuple[str, list[str]]:
    W = instance.warehouse_names
    N = instance.nodes
    deploy = variables["deploy"]
    x = variables["x"]

    selected = [w for w in W if value(deploy[w, drone_name]) and value(deploy[w, drone_name]) > 0.5]
    if not selected:
        return "", []

    depot = selected[0]
    route = [depot]
    current = depot
    visited = set()

    for _ in range(len(instance.customer_names) + 2):
        next_nodes = [
            j
            for j in N
            if j != current
            and value(x[drone_name, current, j])
            and value(x[drone_name, current, j]) > 0.5
        ]
        if not next_nodes:
            break
        nxt = next_nodes[0]
        route.append(nxt)
        if nxt == depot:
            break
        if nxt in visited:
            break
        visited.add(nxt)
        current = nxt

    return depot, route


def solve_milp(instance: Instance, msg: bool = False, time_limit: int | None = 60) -> MilpSolution:
    model, variables = build_model(instance)
    solver = PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    model.solve(solver)

    status = LpStatus[model.status]
    if status != "Optimal":
        return MilpSolution(status, None, {}, {}, {}, {}, {})

    objective = value(model.objective)
    routes: Dict[str, list[str]] = {}
    warehouses: Dict[str, str] = {}
    route_costs: Dict[str, float] = {}
    route_times: Dict[str, float] = {}
    route_loads: Dict[str, float] = {}

    customer = {c.name: c for c in instance.customers}
    drone = {d.name: d for d in instance.drones}

    for d in instance.drone_names:
        depot, route = _reconstruct_route(instance, d, variables)
        if not route:
            continue
        routes[d] = route
        warehouses[d] = depot
        distance = sum(instance.distance[route[i], route[i + 1]] for i in range(len(route) - 1))
        service = sum(customer[n].service_time for n in route if n in customer)
        payload = sum(customer[n].demand for n in route if n in customer)
        deployment = next(w.deployment_cost for w in instance.warehouses if w.name == depot)
        route_costs[d] = distance * drone[d].cost_per_distance + deployment
        route_times[d] = distance + service
        route_loads[d] = payload

    return MilpSolution(status, objective, routes, warehouses, route_costs, route_times, route_loads)
