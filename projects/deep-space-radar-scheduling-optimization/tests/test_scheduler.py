from datetime import timezone
from pathlib import Path

import numpy as np

from orbital_visibility import RadarLocation, SGP4VisibilityEngine, load_tle_records
from radar_scheduler import DeepSpaceRadarScheduler


def test_configuration_uses_integer_slots():
    scheduler = DeepSpaceRadarScheduler()
    assert scheduler.duration_slots == 4
    assert scheduler.min_separation_slots == 12
    assert scheduler.time_slots == 288
    assert scheduler.visibility_mode == "synthetic"


def test_valid_start_requires_full_duration_visibility_and_quality():
    scheduler = DeepSpaceRadarScheduler()
    shape = (len(scheduler.radars), len(scheduler.objects), scheduler.time_slots)
    scheduler.visibility_matrix = np.zeros(shape, dtype=np.int8)
    scheduler.quality_matrix = np.zeros(shape, dtype=float)

    scheduler.visibility_matrix[0, 0, 10:14] = 1
    scheduler.quality_matrix[0, 0, 10:14] = 5.0
    assert (0, 0, 10) in scheduler._valid_starts()

    scheduler.quality_matrix[0, 0, 13] = 0.0
    assert (0, 0, 10) not in scheduler._valid_starts()


def test_solution_metrics_count_full_slot_occupancy():
    scheduler = DeepSpaceRadarScheduler()
    schedule = {r: [] for r in range(len(scheduler.radars))}
    schedule[0] = [
        {
            "object_id": 0,
            "start_slot": 0,
            "start_minute_utc": 0,
            "duration_minutes": 20,
            "quality": 1.0,
        }
    ]
    solution = scheduler._build_solution("Optimal", 1.0, schedule)
    assert solution["radar_utilization"][0] == scheduler.duration_slots / scheduler.time_slots
    assert solution["object_coverage"][0]["observations"] == 1


def test_historical_tle_loads_as_real_catalog_record():
    path = Path("data/vanguard1.tle")
    records = load_tle_records(path)
    assert len(records) == 1
    assert records[0].name == "VANGUARD 1"
    assert records[0].line1.startswith("1 00005U")
    assert records[0].line2.startswith("2 00005")


def test_sgp4_geometry_has_expected_shapes_and_finite_ranges():
    engine = SGP4VisibilityEngine.from_file("data/vanguard1.tle")
    start = engine.default_start_time_utc()
    assert start.tzinfo is not None
    assert start.utcoffset() == timezone.utc.utcoffset(start)

    visibility, elevation, range_km = engine.compute_geometry(
        radars=[RadarLocation("Test", 35.4, -116.9, 1000.0)],
        start_time_utc=start,
        slot_minutes=10,
        time_slots=6,
        min_elevation_deg=5.0,
    )

    assert visibility.shape == (1, 1, 6)
    assert elevation.shape == (1, 1, 6)
    assert range_km.shape == (1, 1, 6)
    assert np.all(np.isfinite(elevation))
    assert np.all(np.isfinite(range_km))
    assert np.all(range_km > 0)


def test_scheduler_tle_mode_uses_catalog_objects_and_geometry():
    scheduler = DeepSpaceRadarScheduler(
        tle_file="data/vanguard1.tle",
        horizon_hours=1,
        slot_minutes=10,
        observation_duration_minutes=20,
        min_separation_minutes=20,
        min_observations=1,
    )
    visibility = scheduler.calculate_visibility_matrix()

    assert scheduler.visibility_mode == "sgp4"
    assert len(scheduler.objects) == 1
    assert scheduler.objects[0].name == "VANGUARD 1"
    assert visibility.shape == (3, 1, 6)
    assert scheduler.elevation_matrix_deg is not None
    assert scheduler.range_matrix_km is not None
