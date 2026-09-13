from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt


def plot_schedule(
    assignments: Sequence[Mapping[str, Any]],
    *,
    num_auvs: int | None = None,
):
    """Plot a Gantt-style schedule from extracted AUV task assignments.

    Each assignment must provide ``auv_id``, ``task_id``, ``start_time`` and
    ``end_time``. The function is deliberately presentation-only: it consumes
    solved output and does not reach back into the MILP model.
    """
    required = {"auv_id", "task_id", "start_time", "end_time"}
    normalized: list[dict[str, float | int]] = []

    for index, assignment in enumerate(assignments):
        missing = required.difference(assignment)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Assignment {index} is missing required fields: {names}")

        auv_id = int(assignment["auv_id"])
        task_id = int(assignment["task_id"])
        start = float(assignment["start_time"])
        end = float(assignment["end_time"])
        if auv_id < 0:
            raise ValueError("auv_id must be non-negative")
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
        if start < 0 or end < start:
            raise ValueError("Assignment times must satisfy 0 <= start_time <= end_time")
        normalized.append(
            {
                "auv_id": auv_id,
                "task_id": task_id,
                "start_time": start,
                "end_time": end,
            }
        )

    if num_auvs is not None and num_auvs < 1:
        raise ValueError("num_auvs must be positive when provided")

    observed = sorted({int(item["auv_id"]) for item in normalized})
    if num_auvs is not None:
        if observed and observed[-1] >= num_auvs:
            raise ValueError("num_auvs is smaller than an observed auv_id")
        auv_ids = list(range(num_auvs))
    else:
        auv_ids = observed

    fig_height = max(3.5, 0.75 * max(len(auv_ids), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    colors = plt.cm.tab10(range(max(len(auv_ids), 1)))
    row_for_auv = {auv_id: row for row, auv_id in enumerate(auv_ids)}

    for item in sorted(normalized, key=lambda row: (int(row["auv_id"]), float(row["start_time"]))):
        auv_id = int(item["auv_id"])
        task_id = int(item["task_id"])
        start = float(item["start_time"])
        end = float(item["end_time"])
        row = row_for_auv[auv_id]
        duration = end - start
        ax.barh(
            row,
            duration,
            left=start,
            height=0.55,
            color=colors[row % len(colors)],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(
            start + duration / 2.0,
            row,
            f"T{task_id}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    if auv_ids:
        ax.set_yticks(range(len(auv_ids)), [f"AUV {auv_id}" for auv_id in auv_ids])
    else:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No scheduled tasks", transform=ax.transAxes, ha="center", va="center")

    ax.set_xlabel("Mission time (hours)")
    ax.set_ylabel("Vehicle")
    ax.set_title("AUV Task Schedule")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_solution_schedule(solution: Mapping[str, Any], *, num_auvs: int | None = None):
    """Plot the schedule stored in an ``AUVOptimizer.solve`` result."""
    if "assignments" not in solution:
        raise ValueError("solution must contain an 'assignments' field")
    return plot_schedule(solution["assignments"], num_auvs=num_auvs)
