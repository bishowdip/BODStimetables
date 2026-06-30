"""Download the keyless supplementary datasets: IMD 2019 and LSOA boundaries.

  * IMD 2019 "File 7" (LSOA scores, ranks, deciles) from gov.uk -> data/raw/imd2019.csv
  * LSOA 2011 boundaries (GeoJSON) -> filtered to the WY bbox -> data/raw/lsoa_wy.geojson

The national LSOA file is ~31 MB; we clip it to the West Yorkshire bounding box so
the committed/working copy is small and the spatial join only deals with WY.
"""
from __future__ import annotations

import argparse
import logging

import requests

from src.common import load_config, project_path

log = logging.getLogger("download_supplementary")
UA = {"User-Agent": "st5011cem-research-project"}


def _download(url: str, dest, session) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip (exists): %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
        tmp.rename(dest)
    log.info("downloaded %s (%d bytes)", dest.name, dest.stat().st_size)


def clip_lsoa_to_bbox(src_path, out_path, bbox) -> None:
    import geopandas as gpd

    gdf = gpd.read_file(src_path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    wy = gdf.cx[bbox["min_lon"]:bbox["max_lon"], bbox["min_lat"]:bbox["max_lat"]]
    wy.to_file(out_path, driver="GeoJSON")
    log.info("clipped LSOA to WY: %d of %d polygons -> %s", len(wy), len(gdf), out_path.name)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    src = cfg["sources"]
    raw = project_path(cfg["paths"]["raw"])
    bbox = cfg["region"]["bbox"]
    session = requests.Session()
    session.headers.update(UA)

    _download(src["imd_url"], raw / "imd2019.csv", session)

    lsoa_full = raw / "lsoa_full.geojson"
    _download(src["lsoa_url"], lsoa_full, session)
    out = project_path(src["lsoa_geojson"])
    if not out.exists():
        try:
            clip_lsoa_to_bbox(lsoa_full, out, bbox)
        except ImportError:
            log.warning("geopandas not installed -- leaving full LSOA file; install it to clip")


if __name__ == "__main__":
    main()
