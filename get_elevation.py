#!/usr/bin/env python3
"""Fetch elevation (AW3D30) for a lon/lat bounding box and save as EXR.

Uses Google Earth Engine (GEE) Python API. You must have:
  1. A Google account with Earth Engine access (https://earthengine.google.com/)
  2. Run `earthengine authenticate` once locally (opens browser) or provide a
	 service account + key if running headless.

Dataset: JAXA/ALOS/AW3D30/V3_2 (30m posting). We'll resample to requested
output pixel dimensions using bilinear interpolation and export a grayscale
PNG where black = min elevation within bbox and white = max elevation unless
explicit min/max specified.

This script keeps dependencies minimal (earthengine-api, Pillow). It uses the
client-side `getDownloadURL` pathway for a quick synchronous download (not an
EE batch export) since we are dealing with modest pixel counts.

Example:
  python get_elevation.py \
	  --bbox -123.30,49.10,-122.90,49.35 \
	  --width 1024 --height 768 \
	  --output output/vancouver_dem.png \
	  --verbose

Later we can add: other DEMs, tiling, GeoTIFF export, slope/hillshade, etc.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import requests

try:
    import ee  # type: ignore
except Exception:  # noqa: BLE001
    ee = None  # fallback; we'll error at runtime with helpful message

AW3D30_COLLECTION = "JAXA/ALOS/AW3D30/V4_1"

# Hardcoded Earth Engine project ID.
# Replace the placeholder below with your actual EE-enabled Google Cloud project ID.
# You can find it in the Earth Engine Code Editor (top left) or your GCP console.
EE_PROJECT_ID = "elevation-maps-468521"

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = bbox_str.split(",")
    if len(parts) != 4:
        raise ValueError("BBox must be min_lon,min_lat,max_lon,max_lat")
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("Invalid bbox order or extents")
    return (min_lon, min_lat, max_lon, max_lat)


def log(msg: str, verbose: bool):  # simple logger
    if verbose:
        print(msg, file=sys.stderr)


def ensure_ee_initialized(verbose: bool):
    if ee is None:
        raise RuntimeError(
            "earthengine-api not installed or failed to import. Did you install requirements and authenticate?"
        )
    project_id = EE_PROJECT_ID or os.environ.get("EARTHENGINE_PROJECT")
    if project_id == "REPLACE_WITH_YOUR_PROJECT_ID":  # user hasn't set it
        log(
            "WARNING: EE_PROJECT_ID still placeholder; attempting default initialization (may fail)",
            verbose,
        )
        project_id = None
    try:
        if project_id:
            log(f"Initializing Earth Engine with project={project_id}", verbose)
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()  # attempt legacy / default
        return
    except Exception:  # noqa: BLE001
        # Attempt interactive auth fallback
        log("Attempting interactive authentication...", verbose)
        ee.Authenticate()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()


def fetch_aw3d30_raw(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    verbose: bool,
    scale: float | None = None,
    crs: str | None = None,
) -> tuple[np.ndarray, str]:
    """Single call to Earth Engine to retrieve raw elevation (meters) as float array.

    Returns (array, selected_band_name)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    collection = ee.ImageCollection(AW3D30_COLLECTION)
    band_candidates = ["DSM", "AVE_DSM"]
    selected_band = None
    try:
        first = collection.first()
        existing = set(first.bandNames().getInfo())
        for b in band_candidates:
            if b in existing:
                selected_band = b
                break
    except Exception:  # noqa: BLE001
        pass
    if not selected_band:
        raise RuntimeError("Unable to find DSM band (DSM/AVE_DSM) in collection")
    dem = (
        collection.mosaic()
        .select([selected_band], ["elev"])
        .updateMask(collection.mosaic().select([selected_band]).neq(-9999))
    )
    params = {
        "region": region.toGeoJSONString(),
        "format": "NPY",
        "bands": "elev",
    }
    # Prefer explicit scale (meters) if provided; else use dimensions to control resampling.
    if scale and scale > 0:
        params["scale"] = float(scale)
    else:
        params["dimensions"] = f"{width}x{height}"
    if crs:
        params["crs"] = crs
    url = dem.getDownloadURL(params)

    log(f"Downloading raw elevation NPY from {url}", verbose)

    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    arr = np.load(io.BytesIO(resp.content)).astype(np.float32)
    return arr, selected_band


def _normalize_if_requested(
    arr: np.ndarray, min_elev: float | None, max_elev: float | None
) -> tuple[np.ndarray, float | None, float | None]:
    if min_elev is None or max_elev is None:
        return arr, None, None  # raw
    if math.isclose(max_elev, min_elev):
        max_elev = min_elev + 1.0
    norm = (arr - min_elev) / (max_elev - min_elev)
    return norm.astype(np.float32), min_elev, max_elev


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch elevation (AW3D30) for bbox and save EXR (raw meters or normalized 0-1)"
    )
    p.add_argument(
        "--bbox",
        required=True,
        help="min_lon,min_lat,max_lon,max_lat (WGS84)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels (if only this is given, height is derived from bbox aspect)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels (if only this is given, width is derived from bbox aspect)",
    )
    p.add_argument("--output", required=True, help="Output EXR path")
    p.add_argument(
        "--scale-hint",
        type=float,
        default=None,
        help="Approximate native resolution (meters) for statistics (default 30)",
    )
    p.add_argument(
        "--mpp",
        type=float,
        default=None,
        help="Desired ground resolution (meters per pixel). If set, derives both width and height. Overrides width/height unless both explicitly provided.",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=50_000_000,
        help="Safety cap on total pixels (width*height). Will scale down proportionally if exceeded.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Explicit scale in meters for sampling (overrides width/height/mpp if provided). Approx native AW3D30 is 30.",
    )
    p.add_argument(
        "--zoom",
        type=int,
        default=None,
        help="Web map style zoom level to approximate browser tile resolution (derives scale; typical 0-24).",
    )
    p.add_argument(
        "--web-mercator",
        action="store_true",
        help="Reproject to Web Mercator (EPSG:3857) like the Code Editor map. Use with --zoom for closest visual match.",
    )
    p.add_argument(
        "--min-elev",
        type=float,
        default=None,
        help="Force min elevation for scaling (else auto)",
    )
    p.add_argument(
        "--max-elev",
        type=float,
        default=None,
        help="Force max elevation for scaling (else auto)",
    )
    # No legend / PNG / secondary outputs
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        bbox = parse_bbox(args.bbox)
    except ValueError as e:  # noqa: BLE001
        parser.error(str(e))
        return 2

    ensure_ee_initialized(args.verbose)

    # Derive dimensions
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_extent = max_lon - min_lon
    lat_extent = max_lat - min_lat
    mid_lat = (min_lat + max_lat) / 2.0
    lon_scale = max(1e-9, math.cos(math.radians(mid_lat)))
    horizontal_m = lon_extent * lon_scale * 111_320.0  # approximate meters
    vertical_m = lat_extent * 111_320.0

    # If zoom provided: derive scale (meters/pixel) using Web Mercator formula at mid latitude.
    derived_scale_from_zoom = None
    if args.zoom is not None:
        # Web Mercator initial resolution (pixels at equator for zoom 0 with 256px tile) * cos(lat) / 2^zoom
        # 156543.03392804097 = 2 * pi * 6378137 / 256
        derived_scale_from_zoom = (
            156543.03392804097 * math.cos(math.radians(mid_lat)) / (2**args.zoom)
        )
        if args.verbose:
            log(
                f"Derived scale {derived_scale_from_zoom:.3f} m/px from zoom {args.zoom} at lat {mid_lat:.3f}",
                True,
            )
        # Override any user scale; we'll supply scale to EE so dimensions become optional.
        args.scale = derived_scale_from_zoom

    # 1. If user provided mpp (meters per pixel) and did NOT force both width & height, compute from mpp (unless zoom/scale provided)
    if args.mpp and not (args.width and args.height) and args.scale is None:
        target_mpp = max(0.01, args.mpp)
        width = max(1, int(round(horizontal_m / target_mpp)))
        height = max(1, int(round(vertical_m / target_mpp)))
    # 1b. If a scale (from --scale or derived --zoom) exists and no explicit width/height given, derive them from bbox physical size.
    elif args.scale is not None and args.width is None and args.height is None:
        width = max(1, int(round(horizontal_m / args.scale)))
        height = max(1, int(round(vertical_m / args.scale)))
    # 2. Otherwise fallback to previous logic using aspect ratio
    elif args.width is None and args.height is None:
        default_width = 1024
        aspect = (lon_extent * lon_scale) / max(1e-9, lat_extent)
        width = default_width
        height = max(1, int(round(width / aspect)))
    elif args.width is not None and args.height is None:
        aspect = (lon_extent * lon_scale) / max(1e-9, lat_extent)
        width = args.width
        height = max(1, int(round(width / aspect)))
    elif args.width is None and args.height is not None:
        aspect = (lon_extent * lon_scale) / max(1e-9, lat_extent)
        height = args.height
        width = max(1, int(round(height * aspect)))
    else:
        width = args.width
        height = args.height

    # Enforce max-pixels cap (uniform scale down)
    total_pixels = width * height
    if total_pixels > args.max_pixels:
        scale_factor = (args.max_pixels / total_pixels) ** 0.5
        width = max(1, int(width * scale_factor))
        height = max(1, int(height * scale_factor))
        if args.verbose:
            log(f"Downscaled to {width}x{height} to respect --max-pixels cap", True)

    # Warn if heavy oversampling relative to native (~30 m) beyond 4x in either axis
    native_width = horizontal_m / 30.0
    native_height = vertical_m / 30.0
    oversample_x = width / max(1.0, native_width)
    oversample_y = height / max(1.0, native_height)
    if (oversample_x > 4 or oversample_y > 4) and args.verbose:
        log(
            f"WARNING: Oversampling (~{oversample_x:.1f}x, {oversample_y:.1f}x) above native resolution; image may look soft when zoomed.",
            True,
        )

    if args.verbose:
        log(f"Requested dimensions (pre-download): {width}x{height}", True)

    out = Path(args.output)
    chosen_crs = "EPSG:3857" if args.web_mercator else None
    arr, band = fetch_aw3d30_raw(
        bbox,
        width,
        height,
        args.verbose,
        scale=args.scale,
        crs=chosen_crs,
    )
    # If actual shape differs (when using scale) log correction.
    actual_h, actual_w = arr.shape
    if (actual_w != width or actual_h != height) and args.verbose:
        log(
            f"Server returned {actual_w}x{actual_h} (was {width}x{height} requested); using actual size.",
            True,
        )
        width, height = actual_w, actual_h

    # If user supplied BOTH min and max, normalize; else keep raw.
    normalized, used_min, used_max = _normalize_if_requested(
        arr, args.min_elev, args.max_elev
    )
    if used_min is not None:
        data_to_save = normalized
        mode_desc = f"normalized 0-1 (min={used_min}, max={used_max})"
    else:
        data_to_save = arr
        mode_desc = "raw meters"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        success = cv2.imwrite(str(out), data_to_save)
        if not success:
            raise RuntimeError("cv2.imwrite returned False")
        if args.verbose:
            log(
                f"Saved EXR {out} [{mode_desc}] (band={band}, min={data_to_save.min():.3f}, max={data_to_save.max():.3f})",
                True,
            )
    except Exception as e:  # noqa: BLE001
        log(f"OpenEXR write failed ({e})", True)
        _write_pfm(out.with_suffix(".pfm"), data_to_save)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# ---------------------- Helpers (PFM) ---------------------------------- #


def _write_pfm(path: Path, data: np.ndarray) -> None:
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    flipped = np.flipud(data)
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        h, w = flipped.shape
        f.write(f"{w} {h}\n".encode())
        f.write(b"-1.0\n")  # little-endian scale
        flipped.tofile(f)
