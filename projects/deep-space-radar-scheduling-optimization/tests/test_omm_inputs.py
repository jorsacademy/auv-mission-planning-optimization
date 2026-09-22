import csv
import json

from orbital_visibility import SGP4VisibilityEngine


ISS_OMM = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2024-05-06T19:53:04.999776",
    "MEAN_MOTION": 15.50957674,
    "ECCENTRICITY": 0.000358,
    "INCLINATION": 51.6393,
    "RA_OF_ASC_NODE": 160.4574,
    "ARG_OF_PERICENTER": 140.6673,
    "MEAN_ANOMALY": 205.725,
    "EPHEMERIS_TYPE": 0,
    "CLASSIFICATION_TYPE": "U",
    "NORAD_CAT_ID": 25544,
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 45212,
    "BSTAR": 0.0002731,
    "MEAN_MOTION_DOT": 0.00015698,
    "MEAN_MOTION_DDOT": 0,
}


def test_load_omm_json(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([ISS_OMM]), encoding="utf-8")

    engine = SGP4VisibilityEngine.from_file(path)

    assert engine.source_format == "omm-json"
    assert engine.records[0].catalog_id == 25544
    assert engine.records[0].name == "ISS (ZARYA)"


def test_load_omm_csv(tmp_path):
    path = tmp_path / "catalog.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISS_OMM))
        writer.writeheader()
        writer.writerow(ISS_OMM)

    engine = SGP4VisibilityEngine.from_file(path)

    assert engine.source_format == "omm-csv"
    assert engine.records[0].catalog_id == 25544
    assert engine.records[0].name == "ISS (ZARYA)"
