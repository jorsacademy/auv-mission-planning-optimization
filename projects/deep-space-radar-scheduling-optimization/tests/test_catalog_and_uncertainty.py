from catalog_client import _metadata_from_row, metadata_by_catalog_id
from uncertainty import evaluate_schedule_uncertainty


def test_satcat_metadata_parses_rcs_and_catalog_id():
    row = {
        "NORAD_CAT_ID": "25544",
        "OBJECT_NAME": "ISS (ZARYA)",
        "OBJECT_ID": "1998-067A",
        "OBJECT_TYPE": "PAY",
        "OWNER": "ISS",
        "OPS_STATUS_CODE": "+",
        "RCS": "399.0524",
        "APOGEE": "421",
        "PERIGEE": "417",
        "INCLINATION": "51.64",
    }
    item = _metadata_from_row(row)
    assert item.norad_cat_id == 25544
    assert item.rcs_m2 == 399.0524
    assert metadata_by_catalog_id([item])[25544].object_name == "ISS (ZARYA)"


def test_satcat_metadata_allows_missing_rcs():
    row = {
        "NORAD_CAT_ID": "1",
        "OBJECT_NAME": "TEST",
        "OBJECT_ID": "",
        "OBJECT_TYPE": "UNK",
        "OWNER": "",
        "OPS_STATUS_CODE": "",
        "RCS": "N/A",
        "APOGEE": "",
        "PERIGEE": "",
        "INCLINATION": "",
    }
    assert _metadata_from_row(row).rcs_m2 is None


def test_uncertainty_rejects_invalid_probability():
    try:
        evaluate_schedule_uncertainty(lambda: object(), simulations=1, visibility_dropout_probability=1.0)
    except ValueError as exc:
        assert "visibility_dropout_probability" in str(exc)
    else:
        raise AssertionError("expected ValueError")
