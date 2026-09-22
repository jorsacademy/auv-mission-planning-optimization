from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math


@dataclass(frozen=True)
class Warehouse:
    name: str
    x: float
    y: float
    deployment_cost: float


@dataclass(frozen=True)
class Customer:
    name: str
    x: float
    y: float
    demand: float
    service_time: float
    earliest: float
    latest: float


@dataclass(frozen=True)
class Drone:
    name: str
    capacity: float
    max_route_time: float
    cost_per_distance: float


@dataclass
class Instance:
    warehouses: List[Warehouse]
    customers: List[Customer]
    drones: List[Drone]
    distance: Dict[Tuple[str, str], float]
    travel_time: Dict[Tuple[str, str], float]

    @property
    def warehouse_names(self) -> List[str]:
        return [w.name for w in self.warehouses]

    @property
    def customer_names(self) -> List[str]:
        return [c.name for c in self.customers]

    @property
    def drone_names(self) -> List[str]:
        return [d.name for d in self.drones]

    @property
    def nodes(self) -> List[str]:
        return self.warehouse_names + self.customer_names


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_instance(data_dir: str | Path) -> Instance:
    data_dir = Path(data_dir)

    warehouses = [
        Warehouse(
            name=row["name"],
            x=float(row["x"]),
            y=float(row["y"]),
            deployment_cost=float(row["deployment_cost"]),
        )
        for row in _read_csv(data_dir / "warehouses.csv")
    ]

    customers = [
        Customer(
            name=row["name"],
            x=float(row["x"]),
            y=float(row["y"]),
            demand=float(row["demand"]),
            service_time=float(row["service_time"]),
            earliest=float(row["earliest"]),
            latest=float(row["latest"]),
        )
        for row in _read_csv(data_dir / "customers.csv")
    ]

    drones = [
        Drone(
            name=row["name"],
            capacity=float(row["capacity"]),
            max_route_time=float(row["max_route_time"]),
            cost_per_distance=float(row["cost_per_distance"]),
        )
        for row in _read_csv(data_dir / "drones.csv")
    ]

    coords = {w.name: (w.x, w.y) for w in warehouses}
    coords.update({c.name: (c.x, c.y) for c in customers})

    distance: Dict[Tuple[str, str], float] = {}
    travel_time: Dict[Tuple[str, str], float] = {}
    for i, (xi, yi) in coords.items():
        for j, (xj, yj) in coords.items():
            if i == j:
                continue
            d = math.hypot(xi - xj, yi - yj)
            distance[i, j] = round(d, 3)
            travel_time[i, j] = round(d, 3)

    return Instance(
        warehouses=warehouses,
        customers=customers,
        drones=drones,
        distance=distance,
        travel_time=travel_time,
    )
