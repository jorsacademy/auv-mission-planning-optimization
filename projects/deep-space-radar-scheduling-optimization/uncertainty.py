from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RobustnessSummary:
    simulations: int
    mean_coverage_pct: float
    std_coverage_pct: float
    p05_coverage_pct: float
    p95_coverage_pct: float
    mean_observations: float
    mean_objective: float
    success_rate_pct: float


def evaluate_schedule_uncertainty(
    scheduler_factory: Callable[[], object],
    *,
    simulations: int = 50,
    visibility_dropout_probability: float = 0.10,
    quality_relative_sigma: float = 0.10,
    seed: int = 42,
    time_limit_seconds: int = 30,
) -> RobustnessSummary:
    """Re-solve the schedule under perturbed visibility and measurement quality.

    Each simulation independently drops visible radar/object/time opportunities and
    perturbs quality multiplicatively. The factory must return a fresh scheduler
    exposing calculate_visibility_matrix(), calculate_observation_quality(), solve(),
    visibility_matrix, quality_matrix and analyze_results().
    """
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if not 0.0 <= visibility_dropout_probability < 1.0:
        raise ValueError("visibility_dropout_probability must be in [0, 1)")
    if quality_relative_sigma < 0.0:
        raise ValueError("quality_relative_sigma must be non-negative")

    rng = np.random.default_rng(seed)
    coverage: list[float] = []
    observations: list[float] = []
    objectives: list[float] = []
    successes = 0

    baseline = scheduler_factory()
    baseline.calculate_visibility_matrix()
    baseline.calculate_observation_quality()
    base_visibility = baseline.visibility_matrix.copy()
    base_quality = baseline.quality_matrix.copy()

    for _ in range(simulations):
        scheduler = scheduler_factory()
        keep = rng.random(base_visibility.shape) >= visibility_dropout_probability
        scheduler.visibility_matrix = (base_visibility * keep).astype(base_visibility.dtype)
        noise = rng.normal(1.0, quality_relative_sigma, size=base_quality.shape)
        scheduler.quality_matrix = np.maximum(0.0, base_quality * noise)
        result = scheduler.solve(time_limit_seconds=time_limit_seconds)
        summary = scheduler.analyze_results()

        if result["status"] in {"Optimal", "Not Solved"}:
            successes += 1
        coverage.append(float(summary["coverage_percentage"]))
        observations.append(float(summary["total_observations"]))
        objectives.append(float(summary["objective_value"]))

    cov = np.asarray(coverage, dtype=float)
    return RobustnessSummary(
        simulations=simulations,
        mean_coverage_pct=float(np.mean(cov)),
        std_coverage_pct=float(np.std(cov)),
        p05_coverage_pct=float(np.percentile(cov, 5)),
        p95_coverage_pct=float(np.percentile(cov, 95)),
        mean_observations=float(np.mean(observations)),
        mean_objective=float(np.mean(objectives)),
        success_rate_pct=100.0 * successes / simulations,
    )
