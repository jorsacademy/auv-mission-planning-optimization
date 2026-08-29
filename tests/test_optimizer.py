import matplotlib
import numpy as np
import pytest

from auv_optimizer import AUVOptimizer

matplotlib.use("Agg")


def make_feasible_optimizer() -> AUVOptimizer:
    optimizer = AUVOptimizer(num_auvs=2, num_tasks=4, seed=7)
    optimizer.auvs.at[0, "max_depth"] = 1000.0
    optimizer.auvs.at[1, "max_depth"] = 1000.0
    optimizer.auvs.at[0, "sensor_types"] = list(optimizer.SENSOR_TYPES)
    optimizer.auvs.at[1, "sensor_types"] = list(optimizer.SENSOR_TYPES)
    optimizer.auvs.at[0, "battery_capacity"] = 100.0
    optimizer.auvs.at[1, "battery_capacity"] = 100.0
    optimizer.calculate_travel_metrics()
    return optimizer


def test_generation_is_reproducible():
    first = AUVOptimizer(num_auvs=3, num_tasks=5, seed=123)
    second = AUVOptimizer(num_auvs=3, num_tasks=5, seed=123)
    assert np.allclose(first.tasks["x_coord"], second.tasks["x_coord"])
    assert first.tasks["required_sensor"].tolist() == second.tasks["required_sensor"].tolist()
    assert first.auvs["sensor_types"].tolist() == second.auvs["sensor_types"].tolist()


def test_travel_metrics_are_auv_specific():
    optimizer = make_feasible_optimizer()
    optimizer.auvs.at[0, "speed"] = 2.0
    optimizer.auvs.at[1, "speed"] = 5.0
    optimizer.calculate_travel_metrics()
    assert optimizer.travel_times.shape == (2, 5, 5)
    assert optimizer.travel_times[0, 0, 1] != optimizer.travel_times[1, 0, 1]


def test_every_task_is_assigned_exactly_once():
    optimizer = make_feasible_optimizer()
    solution = optimizer.solve()
    assert solution is not None
    task_ids = [assignment["task_id"] for assignment in solution["assignments"]]
    assert sorted(task_ids) == list(range(optimizer.num_tasks))
    assert len(task_ids) == len(set(task_ids))


def test_routes_start_and_end_at_depot():
    optimizer = make_feasible_optimizer()
    solution = optimizer.solve()
    assert solution is not None
    assert solution["routes"]
    for route in solution["routes"]:
        assert route["route"][0] == 0
        assert route["route"][-1] == 0


def test_incompatible_task_is_rejected_before_solving():
    optimizer = AUVOptimizer(num_auvs=1, num_tasks=1, seed=4)
    optimizer.auvs.at[0, "sensor_types"] = []
    with pytest.raises(ValueError, match="without a compatible AUV"):
        optimizer.build_optimization_model()


def test_visualization_returns_figure():
    optimizer = make_feasible_optimizer()
    assert optimizer.solve() is not None
    figure = optimizer.visualize_solution()
    assert figure is not None
