# Deep Space Radar Scheduling Optimization

Educational mixed-integer optimization for scheduling radar observations of resident space objects (RSOs), with synthetic visibility plus real SGP4 orbital propagation from modern OMM or legacy TLE data.

The project modernizes an older prototype by making fixed observation duration, separation, energy accounting and coverage logic mutually consistent. It also includes a public-data pipeline for refreshing CelesTrak GP orbital elements and SATCAT metadata, plus uncertainty-aware robustness analysis.

## Features

- MILP scheduling with fixed-duration observation starts.
- Full-slot radar occupancy and sliding-window observation separation.
- Synthetic visibility mode for deterministic experimentation.
- SGP4/topocentric visibility using OMM JSON, OMM CSV, or legacy TLE input.
- Current CelesTrak GP download utility; OMM JSON is the default.
- SATCAT metadata ingestion including radar cross section (RCS) when published.
- Monte Carlo robustness evaluation for visibility and quality uncertainty.
- Schedule visualization and structural/regression tests.

> This is an educational/research prototype, not an operational space-domain-awareness system. Public catalog data can be stale or incomplete, radar physics are simplified, and many operational effects are not modeled.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

CBC is used through PuLP. Most PuLP installations include a compatible CBC binary; otherwise install CBC separately.

## Quick start

Synthetic demonstration:

```bash
python run_scheduler.py
```

Historical TLE example:

```bash
python run_scheduler.py --orbit-file data/vanguard1.tle
```

Current OMM JSON catalog:

```bash
python refresh_catalog.py --group stations --output-dir data/current/stations
python run_scheduler.py --orbit-file data/current/stations/catalog.json
```

Explicit scheduling horizon:

```bash
python run_scheduler.py \
  --orbit-file data/current/stations/catalog.json \
  --start-utc 2026-08-29T00:00:00Z \
  --horizon-hours 24 \
  --output radar_schedule.png
```

## Orbital input formats

The format-neutral CLI accepts:

| Extension | Input | Status |
|---|---|---|
| `.json` | OMM JSON | Preferred |
| `.csv` | OMM CSV | Preferred |
| `.tle` / `.txt` | Traditional TLE | Legacy compatibility |

Skyfield's `EarthSatellite.from_omm()` is used for OMM JSON/CSV, while traditional TLE records use the standard SGP4 constructor. Both paths feed the same topocentric elevation and slant-range geometry engine.

OMM is preferred for current catalogs because the traditional TLE representation has fixed-width constraints, including its historical five-character satellite-number field, while JSON/CSV OMM can represent larger catalog identifiers and can carry additional precision.

## Refresh current public catalog data

The refresh utility uses CelesTrak GP data for orbital elements and SATCAT for catalog metadata.

Default: OMM JSON

```bash
python refresh_catalog.py --group stations --output-dir data/current/stations
```

OMM CSV:

```bash
python refresh_catalog.py --group stations --format csv --output-dir data/current/stations_csv
```

Legacy TLE:

```bash
python refresh_catalog.py --catnr 25544 --format tle --output-dir data/current/iss_tle
```

A default refresh writes:

```text
data/current/.../
├── catalog.json
└── satcat_metadata.json
```

The metadata file includes object type, owner, apogee, perigee, inclination and `RCS` when CelesTrak publishes it. RCS is unavailable for some objects, so the radar-quality model must retain a documented fallback or uncertainty assumption.

`data/current/` is ignored by Git so transient live-catalog snapshots are not accidentally committed.

## Uncertainty-aware scheduling

`uncertainty.py` repeatedly solves fresh perturbed instances after randomly removing visibility opportunities and perturbing observation quality.

```python
from radar_scheduler import DeepSpaceRadarScheduler
from uncertainty import evaluate_schedule_uncertainty

summary = evaluate_schedule_uncertainty(
    lambda: DeepSpaceRadarScheduler(seed=42),
    simulations=50,
    visibility_dropout_probability=0.10,
    quality_relative_sigma=0.10,
)
print(summary)
```

This is a robustness diagnostic, not a full stochastic-programming formulation.

## Model defaults

| Parameter | Default |
|---|---:|
| Horizon | 24 h |
| Time slot | 5 min |
| Observation duration | 20 min |
| Minimum observations per object | 3 |
| Minimum separation between starts | 60 min |
| Synthetic objects | 25 |
| Radar sites | 3 |

## Tests

```bash
pytest -q
```

Tests cover scheduling invariants, TLE parsing/geometry, OMM JSON/CSV loading, SATCAT parsing, and uncertainty perturbations.

## Project structure

```text
.
├── radar_scheduler.py
├── run_scheduler.py
├── orbital_visibility.py
├── catalog_client.py
├── refresh_catalog.py
├── uncertainty.py
├── requirements.txt
├── data/
│   └── vanguard1.tle
├── tests/
│   ├── test_scheduler.py
│   ├── test_catalog_and_uncertainty.py
│   └── test_omm_inputs.py
├── .gitignore
└── README.md
```

## Architecture

1. `refresh_catalog.py` downloads fresh GP orbital elements and SATCAT metadata.
2. `orbital_visibility.py` auto-detects OMM JSON/CSV or TLE and constructs Skyfield SGP4 satellites.
3. Radar-site geometry is propagated into visibility, elevation and slant-range matrices.
4. `radar_scheduler.py` converts feasible fixed-duration observation starts into a MILP.
5. `uncertainty.py` stress-tests the resulting model under availability and quality perturbations.

## Current limitations

- Radar gain/SNR physics remain simplified.
- SATCAT RCS can be missing and is not itself an uncertainty distribution.
- Slew/setup time, maintenance, calibration, weather correlation, track initiation, handoffs and competing missions are omitted.
- Robustness analysis perturbs visibility and quality independently rather than using orbital covariance or correlated scenarios.
- GP element age matters: meaningful current analysis should use recently refreshed elements and an explicit staleness policy.
- The scheduler constructor retains the historical parameter name `tle_file`; use `run_scheduler.py --orbit-file` for the format-neutral public interface.

## Data-source and propagation notes

CelesTrak GP queries provide several representations of general perturbations data. This repository supports the mainstream TLE, OMM JSON and OMM CSV paths. Skyfield propagates the element sets with SGP4 and converts each satellite position into radar-relative topocentric geometry.

For reproducible research, archive the exact orbital-element snapshot used for an experiment rather than silently replacing it with a newer download.

## License

No license has been selected yet. Add one before distributing or reusing the code outside your intended scope.
