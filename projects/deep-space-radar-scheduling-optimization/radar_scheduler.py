from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import random

import matplotlib.pyplot as plt
import numpy as np
from pulp import LpBinary, LpMaximize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value

from orbital_visibility import RadarLocation, SGP4VisibilityEngine


@dataclass(frozen=True)
class RadarSite:
    name: str
    location: Tuple[float, float]
    dish_diameter_m: float
    peak_power_mw: float
    azimuth_coverage_deg: float
    energy_budget_kwh: float
    idle_power_kw: float
    tracking_power_kw: float
    elevation_m: float = 0.0


@dataclass(frozen=True)
class SpaceObject:
    object_id: int
    name: str
    rcs_m2: float
    priority: float
    object_type: str
    orbital_elements: Dict[str, float]


class DeepSpaceRadarScheduler:
    """MILP radar scheduler with synthetic or TLE/SGP4 orbital visibility.

    When ``tle_file`` is provided, orbital positions are propagated with SGP4
    through Skyfield and converted to topocentric elevation/range at each radar.
    Without a TLE file the original deterministic synthetic demonstration mode is
    retained for lightweight examples and structural tests.
    """

    def __init__(
        self,
        seed: int = 42,
        slot_minutes: int = 5,
        horizon_hours: int = 24,
        min_observations: int = 3,
        min_separation_minutes: int = 60,
        observation_duration_minutes: int = 20,
        tle_file: str | Path | None = None,
        start_time_utc: datetime | None = None,
    ) -> None:
        if observation_duration_minutes % slot_minutes != 0:
            raise ValueError("observation_duration_minutes must be divisible by slot_minutes")
        if min_separation_minutes % slot_minutes != 0:
            raise ValueError("min_separation_minutes must be divisible by slot_minutes")

        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.slot_minutes = slot_minutes
        self.time_slots = horizon_hours * 60 // slot_minutes
        self.min_observations = min_observations
        self.min_separation_slots = min_separation_minutes // slot_minutes
        self.duration_slots = observation_duration_minutes // slot_minutes

        self.wavelength_m = 0.1
        self.snr_threshold_db = 10.0
        self.min_elevation_deg = 5.0

        self.radars = self._initialize_radars()
        self.tle_file = Path(tle_file) if tle_file is not None else None
        self.orbit_engine = SGP4VisibilityEngine.from_file(self.tle_file) if self.tle_file else None
        self.start_time_utc = self._normalize_start_time(start_time_utc)
        self.objects = self._initialize_objects()

        self.visibility_matrix: np.ndarray | None = None
        self.quality_matrix: np.ndarray | None = None
        self.elevation_matrix_deg: np.ndarray | None = None
        self.range_matrix_km: np.ndarray | None = None
        self.solution: Dict | None = None

    @property
    def visibility_mode(self) -> str:
        return "sgp4" if self.orbit_engine is not None else "synthetic"

    def _normalize_start_time(self, start_time_utc: datetime | None) -> datetime | None:
        if self.orbit_engine is None:
            return None
        if start_time_utc is None:
            return self.orbit_engine.default_start_time_utc()
        if start_time_utc.tzinfo is None:
            return start_time_utc.replace(tzinfo=timezone.utc)
        return start_time_utc.astimezone(timezone.utc)

    def _initialize_radars(self) -> List[RadarSite]:
        return [
            RadarSite("Maui_Hawaii", (20.7, -156.3), 34.0, 2.0, 240.0, 2000.0, 50.0, 150.0, 3055.0),
            RadarSite("Millstone_Hill", (42.6, -71.5), 26.0, 1.5, 360.0, 1600.0, 40.0, 120.0, 146.0),
            RadarSite("Goldstone_CA", (35.4, -116.9), 70.0, 0.4, 270.0, 1200.0, 30.0, 80.0, 1000.0),
        ]

    def _initialize_objects(self) -> List[SpaceObject]:
        if self.orbit_engine is not None:
            return [
                SpaceObject(
                    object_id=i,
                    name=record.name,
                    rcs_m2=10.0,
                    priority=1.0,
                    object_type="tle_object",
                    orbital_elements={},
                )
                for i, record in enumerate(self.orbit_engine.records)
            ]

        objects: List[SpaceObject] = []
        for i in range(15):
            objects.append(
                SpaceObject(
                    i,
                    f"SAT_{i:02d}",
                    self.rng.uniform(10, 100),
                    1.0,
                    "satellite",
                    {
                        "semi_major_axis_km": self.rng.uniform(25000, 42000),
                        "eccentricity": self.rng.uniform(0, 0.2),
                        "inclination_deg": self.rng.uniform(0, 180),
                    },
                )
            )
        for i in range(15, 21):
            objects.append(
                SpaceObject(
                    i,
                    f"DEB_{i:02d}",
                    self.rng.uniform(1, 10),
                    1.5,
                    "debris",
                    {
                        "semi_major_axis_km": self.rng.uniform(20000, 35000),
                        "eccentricity": self.rng.uniform(0, 0.3),
                        "inclination_deg": self.rng.uniform(0, 180),
                    },
                )
            )
        for i in range(21, 25):
            objects.append(
                SpaceObject(
                    i,
                    f"CRIT_{i:02d}",
                    self.rng.uniform(50, 200),
                    2.0,
                    "critical",
                    {
                        "semi_major_axis_km": self.rng.uniform(35000, 42000),
                        "eccentricity": self.rng.uniform(0, 0.1),
                        "inclination_deg": self.rng.uniform(0, 30),
                    },
                )
            )
        return objects

    def calculate_visibility_matrix(self) -> np.ndarray:
        if self.orbit_engine is not None:
            assert self.start_time_utc is not None
            radar_locations = [
                RadarLocation(r.name, r.location[0], r.location[1], r.elevation_m)
                for r in self.radars
            ]
            visibility, elevation, range_km = self.orbit_engine.compute_geometry(
                radars=radar_locations,
                start_time_utc=self.start_time_utc,
                slot_minutes=self.slot_minutes,
                time_slots=self.time_slots,
                min_elevation_deg=self.min_elevation_deg,
            )
            self.visibility_matrix = visibility
            self.elevation_matrix_deg = elevation
            self.range_matrix_km = range_km
            return visibility

        n_r, n_o = len(self.radars), len(self.objects)
        visibility = np.zeros((n_r, n_o, self.time_slots), dtype=np.int8)
        mu = 3.986e14

        for r, radar in enumerate(self.radars):
            for o, obj in enumerate(self.objects):
                a_m = obj.orbital_elements["semi_major_axis_km"] * 1000.0
                orbital_period_h = 2 * np.pi * np.sqrt(a_m**3 / mu) / 3600.0
                passes_per_day = max(2, int(24 / orbital_period_h))
                for pass_num in range(passes_per_day):
                    pass_start = int((pass_num * 24 / passes_per_day * 60 / self.slot_minutes) % self.time_slots)
                    pass_duration = self.rng.randint(3, 9)
                    max_elevation = self.rng.uniform(5, 80)
                    azimuth = self.rng.uniform(0, 360)
                    if (
                        max_elevation >= self.min_elevation_deg
                        and azimuth <= radar.azimuth_coverage_deg
                        and self.rng.random() < 0.8
                    ):
                        end = min(pass_start + pass_duration, self.time_slots)
                        visibility[r, o, pass_start:end] = 1

        self.visibility_matrix = visibility
        return visibility

    def calculate_observation_quality(self) -> np.ndarray:
        if self.visibility_matrix is None:
            self.calculate_visibility_matrix()

        n_r, n_o = len(self.radars), len(self.objects)
        quality = np.zeros((n_r, n_o, self.time_slots), dtype=float)
        k_b = 1.380649e-23
        noise_temp_k = 100.0
        bandwidth_hz = 1e5

        for r, radar in enumerate(self.radars):
            antenna_gain = (np.pi * radar.dish_diameter_m / self.wavelength_m) ** 2
            for o, obj in enumerate(self.objects):
                for t in np.flatnonzero(self.visibility_matrix[r, o]):
                    if self.orbit_engine is not None:
                        assert self.range_matrix_km is not None
                        assert self.elevation_matrix_deg is not None
                        range_km = float(self.range_matrix_km[r, o, t])
                        elevation = float(self.elevation_matrix_deg[r, o, t])
                    else:
                        base_range_km = obj.orbital_elements["semi_major_axis_km"] * 0.8
                        range_km = base_range_km * (1 + 0.2 * np.sin(2 * np.pi * t / self.time_slots))
                        elevation = max(
                            self.min_elevation_deg,
                            self.rng.uniform(5, 60) * np.sin(2 * np.pi * t / self.time_slots) ** 2,
                        )

                    range_m = max(range_km, 1.0) * 1000.0
                    snr_linear = (
                        radar.peak_power_mw
                        * 1e6
                        * antenna_gain**2
                        * obj.rcs_m2
                        * self.wavelength_m**2
                    ) / ((4 * np.pi) ** 3 * range_m**4 * k_b * noise_temp_k * bandwidth_hz)
                    if snr_linear <= 0:
                        continue
                    snr_db = 10 * np.log10(snr_linear)
                    if snr_db >= self.snr_threshold_db:
                        quality[r, o, t] = min(
                            100.0,
                            (snr_db - self.snr_threshold_db)
                            * np.sin(np.radians(elevation))
                            / 10.0,
                        )

        self.quality_matrix = quality
        return quality

    def _valid_starts(self) -> List[Tuple[int, int, int]]:
        if self.visibility_matrix is None or self.quality_matrix is None:
            raise RuntimeError("visibility and quality matrices must be calculated first")

        starts: List[Tuple[int, int, int]] = []
        latest_start = self.time_slots - self.duration_slots
        for r in range(len(self.radars)):
            for o in range(len(self.objects)):
                for t in range(latest_start + 1):
                    occupied = slice(t, t + self.duration_slots)
                    if (
                        np.all(self.visibility_matrix[r, o, occupied] == 1)
                        and np.all(self.quality_matrix[r, o, occupied] > 0)
                    ):
                        starts.append((r, o, t))
        return starts

    def solve(self, time_limit_seconds: int = 120, relative_gap: float = 0.01) -> Dict:
        if self.visibility_matrix is None:
            self.calculate_visibility_matrix()
        if self.quality_matrix is None:
            self.calculate_observation_quality()

        valid_starts = self._valid_starts()
        problem = LpProblem("Deep_Space_Radar_Scheduling", LpMaximize)
        x = {key: LpVariable(f"x_{key[0]}_{key[1]}_{key[2]}", cat=LpBinary) for key in valid_starts}
        z = {o: LpVariable(f"z_{o}", cat=LpBinary) for o in range(len(self.objects))}

        problem += lpSum(100.0 * self.objects[o].priority * z[o] for o in z) + lpSum(
            0.01 * self.objects[o].priority * self._start_quality(r, o, t) * var
            for (r, o, t), var in x.items()
        )

        for o in range(len(self.objects)):
            obs = [var for (_r2, o2, _t2), var in x.items() if o2 == o]
            if not obs:
                problem += z[o] == 0
                continue
            count = lpSum(obs)
            problem += count >= self.min_observations * z[o]
            problem += count <= (self.min_observations - 1) + len(obs) * z[o]

        for r in range(len(self.radars)):
            for tau in range(self.time_slots):
                occupying = [
                    var
                    for (r2, _o2, t), var in x.items()
                    if r2 == r and t <= tau < t + self.duration_slots
                ]
                if occupying:
                    problem += lpSum(occupying) <= 1

        for o in range(len(self.objects)):
            for window_start in range(self.time_slots):
                vars_in_window = [
                    var
                    for (_r2, o2, t), var in x.items()
                    if o2 == o and window_start <= t < window_start + self.min_separation_slots
                ]
                if vars_in_window:
                    problem += lpSum(vars_in_window) <= 1

        horizon_hours = self.time_slots * self.slot_minutes / 60.0
        obs_hours = self.duration_slots * self.slot_minutes / 60.0
        for r, radar in enumerate(self.radars):
            starts_r = [var for (r2, _o2, _t), var in x.items() if r2 == r]
            idle_energy = radar.idle_power_kw * horizon_hours
            incremental_tracking = max(radar.tracking_power_kw - radar.idle_power_kw, 0.0) * obs_hours
            problem += idle_energy + incremental_tracking * lpSum(starts_r) <= radar.energy_budget_kwh

        problem.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds, gapRel=relative_gap))
        status = LpStatus[problem.status]

        schedule = {r: [] for r in range(len(self.radars))}
        if status in {"Optimal", "Not Solved"}:
            for (r, o, t), var in x.items():
                if value(var) is not None and value(var) > 0.5:
                    schedule[r].append(
                        {
                            "object_id": o,
                            "object_name": self.objects[o].name,
                            "start_slot": t,
                            "start_minute_utc": t * self.slot_minutes,
                            "duration_minutes": self.duration_slots * self.slot_minutes,
                            "quality": self._start_quality(r, o, t),
                        }
                    )
            for r in schedule:
                schedule[r].sort(key=lambda row: row["start_slot"])

        solution = self._build_solution(status, value(problem.objective), schedule)
        self.solution = solution
        return solution

    def _start_quality(self, r: int, o: int, t: int) -> float:
        assert self.quality_matrix is not None
        return float(np.mean(self.quality_matrix[r, o, t : t + self.duration_slots]))

    def _build_solution(self, status: str, objective_value: float | None, schedule: Dict[int, List[Dict]]) -> Dict:
        radar_utilization = {}
        object_coverage = {}
        energy = {}
        horizon_hours = self.time_slots * self.slot_minutes / 60.0
        obs_hours = self.duration_slots * self.slot_minutes / 60.0

        for r, radar in enumerate(self.radars):
            occupied_slots = len(schedule[r]) * self.duration_slots
            radar_utilization[r] = occupied_slots / self.time_slots
            idle_energy = radar.idle_power_kw * horizon_hours
            incremental = max(radar.tracking_power_kw - radar.idle_power_kw, 0.0) * obs_hours * len(schedule[r])
            energy[r] = idle_energy + incremental

        for o in range(len(self.objects)):
            observations = sum(obs["object_id"] == o for rows in schedule.values() for obs in rows)
            object_coverage[o] = {
                "observations": int(observations),
                "meets_minimum": observations >= self.min_observations,
            }

        return {
            "status": status,
            "visibility_mode": self.visibility_mode,
            "start_time_utc": self.start_time_utc.isoformat() if self.start_time_utc else None,
            "objective_value": float(objective_value or 0.0),
            "schedule": schedule,
            "radar_utilization": radar_utilization,
            "radar_energy_kwh": energy,
            "object_coverage": object_coverage,
        }

    def analyze_results(self) -> Dict:
        if self.solution is None:
            raise RuntimeError("solve() must be called first")
        covered = sum(v["meets_minimum"] for v in self.solution["object_coverage"].values())
        total_observations = sum(len(rows) for rows in self.solution["schedule"].values())
        return {
            "status": self.solution["status"],
            "visibility_mode": self.solution["visibility_mode"],
            "start_time_utc": self.solution["start_time_utc"],
            "total_observations": total_observations,
            "coverage_percentage": 100.0 * covered / len(self.objects),
            "average_utilization_percentage": 100.0 * float(np.mean(list(self.solution["radar_utilization"].values()))),
            "objective_value": self.solution["objective_value"],
        }

    def visualize_schedule(self, output: str | Path = "radar_schedule.png", show: bool = False) -> Path:
        if self.solution is None:
            raise RuntimeError("solve() must be called first")
        output = Path(output)
        fig, ax = plt.subplots(figsize=(14, 5))
        for r, radar in enumerate(self.radars):
            for obs in self.solution["schedule"][r]:
                start_h = obs["start_minute_utc"] / 60.0
                duration_h = obs["duration_minutes"] / 60.0
                ax.barh(r, duration_h, left=start_h, height=0.55)
        ax.set_yticks(range(len(self.radars)))
        ax.set_yticklabels([radar.name for radar in self.radars])
        ax.set_xlabel("Hours from scheduling horizon start")
        ax.set_title(f"Radar observation schedule ({self.visibility_mode})")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return output


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-space radar scheduling optimization")
    parser.add_argument("--tle-file", type=Path, help="TLE catalog; enables SGP4 visibility")
    parser.add_argument("--start-utc", type=_parse_utc, help="Horizon start, e.g. 2026-08-29T00:00:00Z")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("radar_schedule.png"))
    args = parser.parse_args()

    scheduler = DeepSpaceRadarScheduler(
        tle_file=args.tle_file,
        start_time_utc=args.start_utc,
        horizon_hours=args.horizon_hours,
    )
    scheduler.solve()
    summary = scheduler.analyze_results()
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
