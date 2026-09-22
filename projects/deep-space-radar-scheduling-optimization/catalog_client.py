from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_SATCAT_URL = "https://celestrak.org/satcat/records.php"
USER_AGENT = "deep-space-radar-scheduling-optimization/1.0"
SUPPORTED_GP_FORMATS = {"tle": "TLE", "json": "JSON", "csv": "CSV"}


@dataclass(frozen=True)
class CatalogMetadata:
    norad_cat_id: int
    object_name: str
    object_id: str | None
    object_type: str
    owner: str | None
    ops_status_code: str | None
    rcs_m2: float | None
    apogee_km: float | None
    perigee_km: float | None
    inclination_deg: float | None


def _http_get_text(base_url: str, params: dict[str, str | int], timeout: float = 30.0) -> str:
    url = f"{base_url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def download_current_gp(
    output: str | Path,
    *,
    gp_format: str = "json",
    group: str | None = None,
    catnr: int | None = None,
    name: str | None = None,
    timeout: float = 30.0,
) -> Path:
    """Download current CelesTrak GP elements as OMM JSON/CSV or legacy TLE."""
    selectors = [("GROUP", group), ("CATNR", catnr), ("NAME", name)]
    active = [(key, value) for key, value in selectors if value is not None]
    if len(active) != 1:
        raise ValueError("provide exactly one of group, catnr, or name")

    normalized_format = gp_format.lower()
    if normalized_format not in SUPPORTED_GP_FORMATS:
        raise ValueError(f"gp_format must be one of {sorted(SUPPORTED_GP_FORMATS)}")

    key, value = active[0]
    text = _http_get_text(
        CELESTRAK_GP_URL,
        {key: value, "FORMAT": SUPPORTED_GP_FORMATS[normalized_format]},
        timeout,
    )
    if not text.strip() or "No GP data found" in text:
        raise RuntimeError("CelesTrak returned no GP data for the requested selector")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def download_current_tle(
    output: str | Path,
    *,
    group: str | None = None,
    catnr: int | None = None,
    name: str | None = None,
    timeout: float = 30.0,
) -> Path:
    """Compatibility wrapper for legacy TLE downloads."""
    return download_current_gp(
        output,
        gp_format="tle",
        group=group,
        catnr=catnr,
        name=name,
        timeout=timeout,
    )


def fetch_satcat_metadata(
    *,
    group: str | None = None,
    catnr: int | None = None,
    name: str | None = None,
    active_only: bool = False,
    on_orbit_only: bool = True,
    max_results: int | None = None,
    timeout: float = 30.0,
) -> list[CatalogMetadata]:
    """Fetch CelesTrak SATCAT metadata, including RCS when available."""
    selectors = [("GROUP", group), ("CATNR", catnr), ("NAME", name)]
    active = [(key, value) for key, value in selectors if value is not None]
    if len(active) != 1:
        raise ValueError("provide exactly one of group, catnr, or name")

    key, value = active[0]
    params: dict[str, str | int] = {key: value, "FORMAT": "CSV"}
    if active_only:
        params["ACTIVE"] = 1
    if on_orbit_only:
        params["ONORBIT"] = 1
    if max_results is not None:
        params["MAX"] = max_results

    text = _http_get_text(CELESTRAK_SATCAT_URL, params, timeout)
    rows = csv.DictReader(io.StringIO(text))
    return [_metadata_from_row(row) for row in rows]


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.upper() in {"N/A", "NULL", "NONE"}:
        return None
    return float(value)


def _metadata_from_row(row: dict[str, str]) -> CatalogMetadata:
    return CatalogMetadata(
        norad_cat_id=int(row["NORAD_CAT_ID"]),
        object_name=row.get("OBJECT_NAME", "").strip(),
        object_id=(row.get("OBJECT_ID") or "").strip() or None,
        object_type=(row.get("OBJECT_TYPE") or "UNK").strip(),
        owner=(row.get("OWNER") or "").strip() or None,
        ops_status_code=(row.get("OPS_STATUS_CODE") or "").strip() or None,
        rcs_m2=_float_or_none(row.get("RCS")),
        apogee_km=_float_or_none(row.get("APOGEE")),
        perigee_km=_float_or_none(row.get("PERIGEE")),
        inclination_deg=_float_or_none(row.get("INCLINATION")),
    )


def metadata_by_catalog_id(rows: Iterable[CatalogMetadata]) -> dict[int, CatalogMetadata]:
    return {row.norad_cat_id: row for row in rows}


def save_metadata_json(rows: Iterable[CatalogMetadata], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.__dict__ for row in rows]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
