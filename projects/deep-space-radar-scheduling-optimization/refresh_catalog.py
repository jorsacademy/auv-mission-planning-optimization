from __future__ import annotations

import argparse
from pathlib import Path

from catalog_client import download_current_gp, fetch_satcat_metadata, save_metadata_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh CelesTrak GP orbital elements and SATCAT metadata")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--group")
    selector.add_argument("--catnr", type=int)
    selector.add_argument("--name")
    parser.add_argument("--format", choices=["json", "csv", "tle"], default="json")
    parser.add_argument("--output-dir", default="data/current")
    parser.add_argument("--max-results", type=int, default=None)
    args = parser.parse_args()

    kwargs = {"group": args.group, "catnr": args.catnr, "name": args.name}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = {"json": "json", "csv": "csv", "tle": "tle"}[args.format]
    orbit_path = download_current_gp(
        output_dir / f"catalog.{extension}",
        gp_format=args.format,
        **kwargs,
    )
    metadata = fetch_satcat_metadata(max_results=args.max_results, **kwargs)
    metadata_path = save_metadata_json(metadata, output_dir / "satcat_metadata.json")

    with_rcs = sum(row.rcs_m2 is not None for row in metadata)
    print(f"Orbital elements: {orbit_path}")
    print(f"Format: {args.format}")
    print(f"Metadata: {metadata_path}")
    print(f"Objects: {len(metadata)}; RCS available: {with_rcs}")


if __name__ == "__main__":
    main()
