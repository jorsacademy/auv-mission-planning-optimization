from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84


@dataclass(frozen=True)
class TLERecord:
    name: str
    line1: str
    line2: str


@dataclass(frozen=True)
class OrbitRecord:
    name: str
    catalog_id: int
    satellite: EarthSatellite
    source_format: str


@dataclass(frozen=True)
class RadarLocation:
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0


def load_tle_records(path: str | Path) -> list[TLERecord]:
    """Load standard 2-line or 3-line TLE records."""
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    records: list[TLERecord] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("2 "):
                raise ValueError(f"Malformed TLE near line {i + 1}")
            norad = lines[i][2:7].strip() or f"OBJECT_{len(records):03d}"
            records.append(TLERecord(f"NORAD_{norad}", lines[i], lines[i + 1]))
            i += 2
            continue

        if i + 2 >= len(lines):
            raise ValueError(f"Incomplete 3-line TLE record near line {i + 1}")
        name, line1, line2 = lines[i : i + 3]
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            raise ValueError(f"Malformed 3-line TLE record near line {i + 1}")
        records.append(TLERecord(name, line1, line2))
        i += 3

    if not records:
        raise ValueError("No TLE records found")
    return records


def _catalog_id(fields: Mapping[str, object]) -> int:
    value = fields.get("NORAD_CAT_ID")
    if value is None:
        raise ValueError("OMM record is missing NORAD_CAT_ID")
    return int(str(value).strip())


def _object_name(fields: Mapping[str, object], catalog_id: int) -> str:
    name = str(fields.get("OBJECT_NAME") or "").strip()
    return name or f"NORAD_{catalog_id}"


def _load_omm_fields(path: Path) -> tuple[list[dict[str, object]], str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError("OMM JSON must contain an object or an array of objects")
        return [dict(row) for row in payload], "omm-json"

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if not rows:
            raise ValueError("No OMM CSV records found")
        return rows, "omm-csv"

    raise ValueError(f"Unsupported OMM file extension: {path.suffix}")


class SGP4VisibilityEngine:
    """Topocentric visibility/range computation backed by Skyfield SGP4.

    Input can be traditional TLE or modern OMM JSON/CSV. OMM is preferred for
    current catalogs because it is not constrained by the TLE five-digit catalog
    number field and can carry additional precision.
    """

    def __init__(self, records: Sequence[OrbitRecord]) -> None:
        if not records:
            raise ValueError("At least one orbital record is required")
        self.records = list(records)
        self.satellites = [record.satellite for record in self.records]
        self.ts = self.satellites[0].ts

    @classmethod
    def from_tle_file(cls, path: str | Path) -> "SGP4VisibilityEngine":
        ts = load.timescale(builtin=True)
        records: list[OrbitRecord] = []
        for tle in load_tle_records(path):
            satellite = EarthSatellite(tle.line1, tle.line2, tle.name, ts)
            records.append(
                OrbitRecord(
                    name=tle.name,
                    catalog_id=int(satellite.model.satnum),
                    satellite=satellite,
                    source_format="tle",
                )
            )
        return cls(records)

    @classmethod
    def from_omm_file(cls, path: str | Path) -> "SGP4VisibilityEngine":
        path = Path(path)
        fields_list, source_format = _load_omm_fields(path)
        ts = load.timescale(builtin=True)
        records: list[OrbitRecord] = []
        for fields in fields_list:
            catalog_id = _catalog_id(fields)
            name = _object_name(fields, catalog_id)
            satellite = EarthSatellite.from_omm(ts, fields)
            satellite.name = name
            records.append(OrbitRecord(name, catalog_id, satellite, source_format))
        return cls(records)

    @classmethod
    def from_file(cls, path: str | Path) -> "SGP4VisibilityEngine":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in {".json", ".csv"}:
            return cls.from_omm_file(path)
        if suffix in {".tle", ".txt"}:
            return cls.from_tle_file(path)
        raise ValueError("Orbit file must be .tle/.txt, .json, or .csv")

    @property
    def source_format(self) -> str:
        formats = {record.source_format for record in self.records}
        return formats.pop() if len(formats) == 1 else "mixed"

    def default_start_time_utc(self) -> datetime:
        epochs = [sat.epoch.utc_datetime().astimezone(timezone.utc) for sat in self.satellites]
        return max(epochs)

    def compute_geometry(
        self,
        radars: Sequence[RadarLocation],
        start_time_utc: datetime,
        slot_minutes: int,
        time_slots: int,
        min_elevation_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if start_time_utc.tzinfo is None:
            start_time_utc = start_time_utc.replace(tzinfo=timezone.utc)
        else:
            start_time_utc = start_time_utc.astimezone(timezone.utc)

        datetimes = [
            start_time_utc + timedelta(minutes=slot_minutes * slot)
            for slot in range(time_slots)
        ]
        times = self.ts.from_datetimes(datetimes)

        shape = (len(radars), len(self.satellites), time_slots)
        visibility = np.zeros(shape, dtype=np.int8)
        elevation = np.zeros(shape, dtype=float)
        range_km = np.zeros(shape, dtype=float)

        for r, radar in enumerate(radars):
            observer = wgs84.latlon(
                radar.latitude_deg,
                radar.longitude_deg,
                elevation_m=radar.elevation_m,
            )
            for o, satellite in enumerate(self.satellites):
                difference = satellite - observer
                alt, _az, distance = difference.at(times).altaz()
                alt_deg = np.asarray(alt.degrees, dtype=float)
                distance_km = np.asarray(distance.km, dtype=float)
                elevation[r, o, :] = alt_deg
                range_km[r, o, :] = distance_km
                visibility[r, o, :] = (alt_deg >= min_elevation_deg).astype(np.int8)

        return visibility, elevation, range_km
