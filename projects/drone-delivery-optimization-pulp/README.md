# Drone Delivery Optimization with PuLP

This repository contains an intermediate-to-advanced optimization project for a **multi-depot capacitated drone routing problem with time windows (CVRPTW)**.

The project now includes both an exact Mixed-Integer Linear Programming model and a Hill Climbing baseline so that students can compare mathematical optimization against a local-search heuristic on the same data set.

## Main Features

- Multiple warehouses
- Multiple heterogeneous drones
- Customer demand
- Drone payload capacity
- Maximum route duration
- Customer service times
- Customer time windows
- Warehouse deployment cost
- Drone-specific travel cost
- Customer-to-customer routing
- Flow conservation
- Load-based subtour elimination
- CBC solver through PuLP
- Hill Climbing benchmark
- CSV-based input data
- Route visualization with Matplotlib

## Project Structure

```text
.
├── data/
│   ├── customers.csv
│   ├── drones.csv
│   └── warehouses.csv
├── drone_delivery/
│   ├── __init__.py
│   ├── data.py
│   ├── hill_climbing.py
│   ├── model.py
│   ├── reporting.py
│   └── visualization.py
├── compare_methods.py
├── drone_delivery_optimization.py
├── main.py
├── MODEL.md
├── requirements.txt
└── LICENSE
```

## Optimization Model

The MILP model determines:

1. which drones are activated,
2. which warehouse each active drone uses,
3. which drone serves each customer,
4. which directed arcs are used in every drone route,
5. arrival time at each served customer,
6. cumulative payload after visiting each customer.

The objective minimizes total routing cost plus warehouse deployment cost.

The model includes customer coverage, depot consistency, flow conservation, payload capacity, route-duration, time-window, return-deadline, and subtour-elimination constraints.

A detailed mathematical formulation is available in [MODEL.md](MODEL.md).

## CSV Input

The optimization data are separated from the code.

### `data/warehouses.csv`

Contains warehouse coordinates and deployment costs.

### `data/customers.csv`

Contains customer coordinates, demand, service time, and time-window bounds.

### `data/drones.csv`

Contains drone payload capacity, maximum route duration, and travel cost coefficient.

Euclidean distances are generated automatically from the coordinates in the CSV files.

## Installation

```bash
pip install -r requirements.txt
```

## Run the MILP Solver

```bash
python main.py
```

The script solves the multi-depot drone routing model with CBC and prints the selected routes. It also generates:

```text
milp_routes.png
```

## Compare MILP and Hill Climbing

```bash
python compare_methods.py
```

This executes both methods on the same instance and reports:

- MILP objective value,
- Hill Climbing objective value,
- route structure for both approaches,
- Hill Climbing iteration count,
- percentage heuristic gap relative to the MILP optimum.

It also creates route plots for both methods.

## Hill Climbing Baseline

The Hill Climbing implementation first constructs a feasible solution using a greedy insertion procedure. It then improves that solution using two neighborhoods:

1. **2-opt reversal** inside a drone route,
2. **customer relocation** between drone routes, including alternative warehouse assignments for the receiving route.

Only feasible moves are accepted. Capacity, customer time windows, and maximum drone route duration are checked before a candidate solution can replace the incumbent solution.

This provides a useful comparison between an exact MILP method and a local-search method. The Hill Climbing result is not guaranteed to be globally optimal.

## Backward-Compatible Entry Point

The original file name is retained:

```bash
python drone_delivery_optimization.py
```

It now calls the modular application in `main.py`.

## Educational Purpose

This project is intended to demonstrate:

- mathematical optimization modeling,
- mixed-integer programming,
- vehicle-routing logic,
- subtour elimination,
- time-window constraints,
- heuristic local search,
- solver benchmarking,
- separation of model, data, reporting, and visualization layers.

## License

This project is distributed under a custom non-commercial license. Commercial use, resale, commercial integration, and commercial redistribution are prohibited without prior written permission from the copyright holder. See [LICENSE](LICENSE) for the complete terms.
