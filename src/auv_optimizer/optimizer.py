from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class ObjectiveWeights:
    mission_time: float = 0.5
    energy: float = 0.3
    priority: float = 0.2


class AUVOptimizer:
    """MILP optimizer for heterogeneous AUV mission planning.

    The model assigns every task exactly once, routes each used AUV from and
    back to the depot, enforces sensor/depth compatibility, respects operating
    time limits, and sequences tasks with time-based subtour elimination.
    """

    SENSOR_TYPES = ("sonar", "camera", "hydrophone", "ctd", "magnetometer")

    def __init__(
        self,
        num_auvs: int = 3,
        num_tasks: int = 10,
        seed: int = 42,
        weights: ObjectiveWeights | None = None,
    ) -> None:
        if num_auvs < 1 or num_tasks < 1:
            raise ValueError("num_auvs and num_tasks must both be positive")
        self.num_auvs = num_auvs
        self.num_tasks = num_tasks
        self.seed = seed
        self.weights = weights or ObjectiveWeights()
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self.depot = {"x_coord": 0.0, "y_coord": 0.0, "depth": 0.0}
        self.generate_data()

    def generate_data(self) -> None:
        self.auvs = pd.DataFrame(
            {
                "id": range(self.num_auvs),
                "max_depth": self._rng.uniform(200, 1000, self.num_auvs),
                "battery_capacity": self._rng.uniform(8, 24, self.num_auvs),
                "speed": self._rng.uniform(2, 5, self.num_auvs),
                "sensor_types": [
                    self._py_rng.sample(
                        list(self.SENSOR_TYPES), self._py_rng.randint(2, len(self.SENSOR_TYPES))
                    )
                    for _ in range(self.num_auvs)
                ],
            }
        )
        self.tasks = pd.DataFrame(
            {
                "id": range(self.num_tasks),
                "x_coord": self._rng.uniform(0, 10000, self.num_tasks),
                "y_coord": self._rng.uniform(0, 10000, self.num_tasks),
                "depth": self._rng.uniform(50, 800, self.num_tasks),
                "priority": self._rng.integers(1, 6, self.num_tasks),
                "duration": self._rng.uniform(0.5, 4, self.num_tasks),
                "required_sensor": [
                    self._py_rng.choice(self.SENSOR_TYPES) for _ in range(self.num_tasks)
                ],
            }
        )
        self.calculate_travel_metrics()

    def _point(self, idx: int) -> np.ndarray:
        if idx == 0:
            return np.array([0.0, 0.0, 0.0])
        row = self.tasks.loc[idx - 1]
        return np.array([row.x_coord, row.y_coord, row.depth], dtype=float)

    def calculate_travel_metrics(self) -> None:
        n = self.num_tasks + 1
        self.distances = np.zeros((n, n), dtype=float)
        self.travel_times = np.zeros((self.num_auvs, n, n), dtype=float)
        self.energy_consumption = np.zeros((self.num_auvs, n, n), dtype=float)

        current_direction = np.array([1.0, 0.5, 0.0])
        current_direction /= np.linalg.norm(current_direction)
        current_strength = 0.5

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                delta = self._point(j) - self._point(i)
                distance = float(np.linalg.norm(delta))
                self.distances[i, j] = distance
                direction = delta / distance if distance else np.zeros(3)
                current_effect = float(np.dot(current_direction, direction) * current_strength)

                for a in range(self.num_auvs):
                    speed = float(self.auvs.loc[a, "speed"])
                    effective_speed = max(0.5, speed + current_effect)
                    travel_time = distance / (effective_speed * 1852.0)
                    # Relative energy proxy; kept separate from battery-time capacity.
                    energy = distance * (1.0 + 0.2 * (speed / effective_speed) ** 2)
                    self.travel_times[a, i, j] = travel_time
                    self.energy_consumption[a, i, j] = energy

    def check_compatibility(self, auv_id: int, task_id: int) -> bool:
        required_sensor = self.tasks.loc[task_id, "required_sensor"]
        task_depth = float(self.tasks.loc[task_id, "depth"])
        sensors = self.auvs.loc[auv_id, "sensor_types"]
        max_depth = float(self.auvs.loc[auv_id, "max_depth"])
        return required_sensor in sensors and task_depth <= max_depth

    def _validate_feasibility(self) -> None:
        impossible = [
            task
            for task in range(self.num_tasks)
            if not any(self.check_compatibility(a, task) for a in range(self.num_auvs))
        ]
        if impossible:
            raise ValueError(f"Tasks without a compatible AUV: {impossible}")

    def build_optimization_model(self) -> LpProblem:
        self._validate_feasibility()
        n = self.num_tasks + 1
        model = LpProblem("AUV_Fleet_Optimization", LpMinimize)

        self.x = LpVariable.dicts(
            "route",
            [(a, i, j) for a in range(self.num_auvs) for i in range(n) for j in range(n) if i != j],
            cat=LpBinary,
        )
        self.y = LpVariable.dicts(
            "assignment",
            [(a, i) for a in range(self.num_auvs) for i in range(1, n)],
            cat=LpBinary,
        )
        self.used = LpVariable.dicts("used", range(self.num_auvs), cat=LpBinary)
        self.start_time = LpVariable.dicts(
            "start_time", [(a, i) for a in range(self.num_auvs) for i in range(1, n)], lowBound=0
        )
        self.mission_time = LpVariable("mission_time", lowBound=0)

        total_energy = lpSum(
            self.energy_consumption[a, i, j] * self.x[a, i, j]
            for a in range(self.num_auvs)
            for i in range(n)
            for j in range(n)
            if i != j
        )
        priority_reward = lpSum(
            float(self.tasks.loc[i - 1, "priority"]) * self.y[a, i]
            for a in range(self.num_auvs)
            for i in range(1, n)
        )
        w = self.weights
        model += w.mission_time * self.mission_time + w.energy * total_energy - w.priority * priority_reward

        for i in range(1, n):
            model += lpSum(self.y[a, i] for a in range(self.num_auvs)) == 1

        for a in range(self.num_auvs):
            model += lpSum(self.x[a, 0, j] for j in range(1, n)) == self.used[a]
            model += lpSum(self.x[a, i, 0] for i in range(1, n)) == self.used[a]
            model += lpSum(self.y[a, i] for i in range(1, n)) >= self.used[a]
            model += lpSum(self.y[a, i] for i in range(1, n)) <= self.num_tasks * self.used[a]

            for i in range(1, n):
                model += lpSum(self.x[a, j, i] for j in range(n) if j != i) == self.y[a, i]
                model += lpSum(self.x[a, i, j] for j in range(n) if j != i) == self.y[a, i]
                if not self.check_compatibility(a, i - 1):
                    model += self.y[a, i] == 0

            operating_time = lpSum(
                float(self.tasks.loc[i - 1, "duration"]) * self.y[a, i] for i in range(1, n)
            ) + lpSum(
                self.travel_times[a, i, j] * self.x[a, i, j]
                for i in range(n)
                for j in range(n)
                if i != j
            )
            model += operating_time <= float(self.auvs.loc[a, "battery_capacity"])

        # A data-derived M keeps the time formulation tighter than an arbitrary constant.
        max_task = float(self.tasks["duration"].sum())
        max_leg = float(self.travel_times.max())
        big_m = max_task + (self.num_tasks + 1) * max_leg + 1.0

        for a in range(self.num_auvs):
            for j in range(1, n):
                model += self.start_time[a, j] >= self.travel_times[a, 0, j] - big_m * (1 - self.x[a, 0, j])
            for i in range(1, n):
                duration_i = float(self.tasks.loc[i - 1, "duration"])
                for j in range(1, n):
                    if i == j:
                        continue
                    model += self.start_time[a, j] >= (
                        self.start_time[a, i]
                        + duration_i
                        + self.travel_times[a, i, j]
                        - big_m * (1 - self.x[a, i, j])
                    )
                model += self.mission_time >= (
                    self.start_time[a, i]
                    + duration_i
                    + self.travel_times[a, i, 0]
                    - big_m * (1 - self.x[a, i, 0])
                )

        self.model = model
        return model

    def solve(self, msg: bool = False) -> dict[str, Any] | None:
        if not hasattr(self, "model"):
            self.build_optimization_model()
        self.model.solve(PULP_CBC_CMD(msg=msg))
        if LpStatus[self.model.status] != "Optimal":
            return None
        return self.extract_solution()

    def extract_solution(self) -> dict[str, Any]:
        routes: list[dict[str, Any]] = []
        assignments: list[dict[str, Any]] = []
        n = self.num_tasks + 1

        for a in range(self.num_auvs):
            if value(self.used[a]) < 0.5:
                continue
            route = [0]
            current = 0
            visited: set[int] = set()
            while True:
                next_node = next(
                    (
                        j
                        for j in range(n)
                        if j != current and value(self.x[a, current, j]) is not None and value(self.x[a, current, j]) > 0.5
                    ),
                    None,
                )
                if next_node is None:
                    raise RuntimeError(f"Broken route extraction for AUV {a}")
                route.append(next_node)
                if next_node == 0:
                    break
                if next_node in visited:
                    raise RuntimeError(f"Subtour detected while extracting AUV {a}")
                visited.add(next_node)
                current = next_node

            routes.append({"auv_id": a, "route": route})
            for task_node in route[1:-1]:
                task_id = task_node - 1
                start = float(value(self.start_time[a, task_node]))
                duration = float(self.tasks.loc[task_id, "duration"])
                assignments.append(
                    {
                        "auv_id": a,
                        "task_id": task_id,
                        "start_time": start,
                        "end_time": start + duration,
                    }
                )

        assignments.sort(key=lambda item: item["start_time"])
        self.solution = {
            "objective_value": float(value(self.model.objective)),
            "mission_time": float(value(self.mission_time)),
            "routes": routes,
            "assignments": assignments,
        }
        return self.solution

    def print_solution_summary(self) -> None:
        if not hasattr(self, "solution"):
            raise RuntimeError("Solve the model before printing a summary")
        print("=== AUV Fleet Mission Summary ===")
        print(f"Total mission time: {self.solution['mission_time']:.2f} hours")
        print(f"Objective value: {self.solution['objective_value']:.2f}")
        print(f"Tasks completed: {len(self.solution['assignments'])}/{self.num_tasks}")
        print(f"AUVs utilized: {len(self.solution['routes'])}/{self.num_auvs}")
        for route_info in self.solution["routes"]:
            labels = ["Depot" if node == 0 else f"Task {node - 1}" for node in route_info["route"]]
            print(f"AUV {route_info['auv_id']}: {' -> '.join(labels)}")

    def visualize_solution(self):
        if not hasattr(self, "solution"):
            raise RuntimeError("Solve the model before visualization")

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            self.tasks["x_coord"],
            self.tasks["y_coord"],
            -self.tasks["depth"],
            c=self.tasks["priority"],
            cmap="viridis",
            s=90,
            alpha=0.7,
        )
        ax.scatter([0], [0], [0], marker="*", s=180, label="Depot")
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, self.num_auvs)))
        for info in self.solution["routes"]:
            a = info["auv_id"]
            points = [self._point(node) for node in info["route"]]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [-p[2] for p in points]
            ax.plot(xs, ys, zs, "o-", c=colors[a], label=f"AUV {a}")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_zlabel("Depth (meters)")
        ax.set_title("AUV Mission Plan")
        fig.colorbar(scatter, ax=ax, pad=0.1, label="Task Priority")
        ax.legend()
        return fig
