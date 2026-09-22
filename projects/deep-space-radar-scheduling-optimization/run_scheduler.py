from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from radar_scheduler import DeepSpaceRadarScheduler


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep-space radar scheduler with synthetic, TLE, or OMM orbital input"
    )
    parser.add_argument(
        "--orbit-file",
        type=Path,
        help="Orbital catalog: .json/.csv OMM (preferred) or .tle/.txt legacy TLE",
    )
    parser.add_argument("--start-utc", type=_parse_utc, help="Horizon start, e.g. 2026-08-29T00:00:00Z")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("radar_schedule.png"))
    args = parser.parse_args()

    scheduler = DeepSpaceRadarScheduler(
        tle_file=args.orbit_file,  # historical constructor name; engine auto-detects format
        start_time_utc=args.start_utc,
        horizon_hours=args.horizon_hours,
    )
    scheduler.solve()
    summary = scheduler.analyze_results()

    source_format = scheduler.orbit_engine.source_format if scheduler.orbit_engine else "synthetic"
    print(f"Orbit source: {source_format}")
    print(f"Visibility mode: {summary['visibility_mode']}")
    if summary["start_time_utc"]:
        print(f"Horizon start: {summary['start_time_utc']}")
    print(f"Status: {summary['status']}")
    print(f"Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"Observations: {summary['total_observations']}")
    print(f"Average utilization: {summary['average_utilization_percentage']:.1f}%")
    print(f"Objective: {summary['objective_value']:.2f}")
    scheduler.visualize_schedule(args.output)


if __name__ == "__main__":
    main()
