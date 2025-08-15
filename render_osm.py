#!/usr/bin/env python3
"""Simple OSM highway + border renderer from PostGIS (pure Python).

Rewrites the prior Overpass API-based script to query a local PostGIS
database with OSM data.

Features:
  * Fetches OSM data from a PostGIS database for a given bbox
  * Highways (motorway, trunk, primary, secondary, tertiary)
  * National borders (admin_level=2)
  * Renders layered, weighted black lines onto a white background
  * Antialiasing via OpenCV LINE_AA + sub-pixel precision (shift=4)

Usage:
  python render_osm.py \
      --bbox min_lon,min_lat,max_lon,max_lat \
      --output output/vancouver_test.png \
      --width 2048 --height 1536 \
      --db-host 192.168.1.3 --db-name gis --db-user osm

Optional flags:
  --no-borders         Skip querying / drawing borders
  --verbose            More progress logging

Dependencies (see requirements.txt):
  opencv-python, numpy, psycopg3

Notes:
  - Assumes an osm2pgsql-style schema (e.g., osm_filtered_data table).
  - Database credentials should be handled securely (e.g., env vars).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psycopg
from psycopg.rows import dict_row

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Database configuration
DB_HOST = "192.168.1.3"
DB_NAME = "osm_db"
DB_USER = "postgres"
DB_PASSWORD = None

HIGHWAY_TAGS = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
]

LAYER_STYLES = {
    "borders": {"thickness": 6, "color": 0},  # Dark gray for borders
    #
    "motorway": {"thickness": 2, "color": 0},  # Very dark gray (main roads)
    "trunk": {"thickness": 1, "color": 0},  # Dark gray
    "primary": {"thickness": 1, "color": 0.2},  # Medium-dark gray
    "secondary": {"thickness": 1, "color": 0.4},  # Medium gray
    "tertiary": {"thickness": 1, "color": 0.8},  # Lighter gray (brighter)
    #
    "water": {"thickness": 2, "color": 0},  # Medium-dark gray for water
    "railways": {"thickness": 1, "color": 0},  # Dark gray for railways
}


class Config:
    def __init__(
        self,
        bbox,
        width,
        height,
        output,
        render_types=None,
        verbose=False,
    ):
        self.bbox = bbox  # (min_lon, min_lat, max_lon, max_lat)
        self.width = width
        self.height = height
        self.output = output if isinstance(output, Path) else Path(output)
        self.render_types = render_types or ["roads", "borders"]
        self.verbose = verbose

        # Check if output is EXR format
        self.is_exr = self.output.suffix.lower() == ".exr"


# ---------------------- Utility / IO ------------------------------------ #


def parse_bbox(bbox_str):
    parts = bbox_str.split(",")
    if len(parts) != 4:
        raise ValueError(
            "BBox must have 4 comma-separated numbers: min_lon,min_lat,max_lon,max_lat"
        )
    min_lon, min_lat, max_lon, max_lat = map(float, parts)
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("Invalid bbox coordinate ordering or extents")
    return (min_lon, min_lat, max_lon, max_lat)


def log(msg, cfg, *, force=False):
    if cfg.verbose or force:
        print(msg, file=sys.stderr)


def log_timing(start_time, operation, cfg):
    """Log timing information in verbose mode"""
    if cfg.verbose:
        elapsed = time.time() - start_time
        if elapsed < 1:
            print(f"[TIMING] {operation}: {elapsed * 1000:.1f}ms", file=sys.stderr)
        else:
            print(f"[TIMING] {operation}: {elapsed:.2f}s", file=sys.stderr)


def fetch_from_postgis(query, cfg):
    start_time = time.time()
    conn_string = f"host={DB_HOST} dbname={DB_NAME} user={DB_USER}"
    if DB_PASSWORD:
        conn_string += f" password={DB_PASSWORD}"

    try:
        log(f"[PostGIS] Connecting to {DB_HOST}...", cfg)
        if cfg.verbose:
            log(f"[PostGIS] Query: {query}", cfg)

        connect_start = time.time()
        with psycopg.connect(conn_string, row_factory=dict_row) as conn:
            log_timing(connect_start, "Database connection", cfg)

            with conn.cursor() as cur:
                log(f"[PostGIS] Executing query ({len(query)} chars)", cfg)

                query_start = time.time()
                cur.execute(query)
                results = cur.fetchall()
                log_timing(query_start, "Query execution", cfg)

                log(f"[PostGIS] Fetched {len(results)} rows", cfg)
                log_timing(start_time, "Total PostGIS fetch", cfg)
                return results
    except psycopg.Error as e:
        raise RuntimeError(f"PostGIS query failed: {e}")


# ---------------------- Query Construction ------------------------------ #


def make_borders_query(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    query = f"""
    SELECT
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."labeled_countries"
    WHERE
        way && ST_Transform(ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326), 3857);
    """
    return query


def make_highways_query(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    highway_filter = ", ".join([f"'{tag}'" for tag in HIGHWAY_TAGS])
    query = f"""
    SELECT
        highway,
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."osm_roads"
    WHERE
        way && ST_Transform(ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326), 3857)
        AND "highway" IN ({highway_filter});
    """
    return query


def make_water_query(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    query = f"""
    SELECT
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."osm_waterways"
    WHERE
        way && ST_Transform(ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326), 3857)
        AND "waterway" = 'river'
        AND ST_Length(way) > 90;
    """
    return query


def make_railways_query(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    query = f"""
    SELECT
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."osm_railways"
    WHERE
        way && ST_Transform(ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326), 3857)
    """
    return query


# ---------------------- Data Extraction --------------------------------- #


def extract_highway_lines(results):
    lines = {tag: [] for tag in HIGHWAY_TAGS}
    for row in results:
        highway_type = row["highway"]
        if highway_type not in lines:
            continue
        geom_json = json.loads(row["geom"])
        coords = geom_json.get("coordinates", [])
        if len(coords) < 2:
            continue
        # GeoJSON is (lon, lat), which is what we want
        lines[highway_type].append(coords)
    return lines


def extract_border_polygons(results):
    polygons = []
    for row in results:
        geom_json = json.loads(row["geom"])
        geom_type = geom_json.get("type", "")
        coords = geom_json.get("coordinates", [])

        if geom_type == "Polygon":
            # Single polygon - take the exterior ring (first coordinate array)
            if coords and len(coords[0]) >= 3:
                polygons.append(coords[0])
        elif geom_type == "MultiPolygon":
            # Multiple polygons - extract exterior ring from each
            for polygon in coords:
                if polygon and len(polygon[0]) >= 3:
                    polygons.append(polygon[0])
    return polygons


def extract_water_lines(results):
    lines = []
    for row in results:
        geom_json = json.loads(row["geom"])
        coords = geom_json.get("coordinates", [])
        if len(coords) < 2:
            continue
        lines.append(coords)
    return lines


def extract_railway_lines(results):
    lines = []
    for row in results:
        geom_json = json.loads(row["geom"])
        coords = geom_json.get("coordinates", [])
        if len(coords) < 2:
            continue
        lines.append(coords)
    return lines


# ---------------------- Rendering --------------------------------------- #


def lonlat_to_pixel_batch(coords, bbox, w, h, shift=4):
    """Vectorized coordinate transformation with sub-pixel precision."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    coords_array = np.array(coords)
    shift_factor = 1 << shift

    # Keep floating point precision for sub-pixel accuracy
    x = (coords_array[:, 0] - min_lon) / lon_range * w * shift_factor
    y = (max_lat - coords_array[:, 1]) / lat_range * h * shift_factor

    # Round before casting to integer
    return np.column_stack((np.round(x).astype(np.int32), np.round(y).astype(np.int32)))


def draw_polygons_opencv(img, polygons, bbox, scaled_w, scaled_h, color, shift=4):
    """Draw filled polygons using OpenCV with sub-pixel precision"""
    total_polygons = 0
    start_time = time.time()
    shift_factor = 1 << shift  # 2^shift

    for polygon in polygons:
        if len(polygon) < 3:  # Need at least 3 points for a polygon
            continue

        # Convert coordinates in batch using numpy with sub-pixel precision
        pts = lonlat_to_pixel_batch(polygon, bbox, scaled_w, scaled_h, shift)

        # Simple culling - skip polygons completely outside image bounds
        if len(pts) > 0:
            min_x, min_y = pts.min(axis=0)
            max_x, max_y = pts.max(axis=0)

            # Convert bounds to shifted coordinate space for comparison
            if (
                max_x < 0
                or min_x > (scaled_w * shift_factor)
                or max_y < 0
                or min_y > (scaled_h * shift_factor)
            ):
                continue

        # Draw filled polygon with OpenCV using sub-pixel precision
        if len(pts) > 2:
            cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA, shift=shift)
            total_polygons += 1

    elapsed = time.time() - start_time
    return total_polygons, elapsed


def draw_lines_opencv(img, lines, bbox, scaled_w, scaled_h, thickness, color, shift=4):
    """Optimized line drawing using OpenCV with sub-pixel precision."""
    total_lines = 0
    start_time = time.time()

    for line in lines:
        if len(line) < 2:
            continue

        # Convert coordinates in batch using numpy with sub-pixel precision
        pts = lonlat_to_pixel_batch(line, bbox, scaled_w, scaled_h, shift)

        if len(pts) > 1:
            cv2.polylines(
                img,
                [pts],
                isClosed=False,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                shift=shift,
            )
            total_lines += 1

    elapsed = time.time() - start_time
    return total_lines, elapsed


def render_single_layer(
    cfg, layer_name, highway_layers, border_polygons, water_lines, railways_lines
):
    """Render a single layer type to its own image"""
    # Create OpenCV image (32-bit float format, white background) for better antialiasing visibility
    img = np.full((cfg.height, cfg.width, 3), 255, dtype=np.uint8)

    # Pre-calculate bbox values for performance
    bbox_cached = cfg.bbox

    # Define a function to convert grayscale value to BGR color
    def grayscale_to_bgr(gray_value):
        """Convert grayscale value (0.0-1.0) to BGR tuple for OpenCV (0-255 range)"""
        gray_255 = int(gray_value * 255)
        return (gray_255, gray_255, gray_255)

    # Draw in order of thickness (smallest first)
    draw_order = ["tertiary", "secondary", "primary", "trunk", "motorway"]

    # Draw the specific layer
    if layer_name == "roads":
        log("[Render] Drawing roads layer...", cfg)
        roads_start = time.time()

        for name in draw_order:
            lines = highway_layers.get(name, [])
            if not lines:
                continue
            # Get thickness and color from layer styles
            layer_style = LAYER_STYLES[name]
            thickness = layer_style["thickness"]
            color = grayscale_to_bgr(layer_style["color"])
            log(
                f"[Render] Drawing {len(lines)} {name} lines (thickness {thickness}px, color {layer_style['color']})",
                cfg,
            )

            layer_start = time.time()
            drawn_count, draw_time = draw_lines_opencv(
                img,
                lines,
                bbox_cached,
                cfg.width,
                cfg.height,
                thickness,
                color,
            )
            log_timing(layer_start, f"Drawing {name} layer ({drawn_count} lines)", cfg)

        log_timing(roads_start, "Total roads rendering", cfg)

    elif layer_name == "water" and water_lines:
        log(f"[Render] Drawing {len(water_lines)} water lines...", cfg)
        layer_style = LAYER_STYLES["water"]
        thickness = layer_style["thickness"]
        color = grayscale_to_bgr(layer_style["color"])

        water_start = time.time()
        drawn_count, draw_time = draw_lines_opencv(
            img,
            water_lines,
            bbox_cached,
            cfg.width,
            cfg.height,
            thickness,
            color,
        )
        log_timing(water_start, f"Water rendering ({drawn_count} lines)", cfg)

    elif layer_name == "railways" and railways_lines:
        log(f"[Render] Drawing {len(railways_lines)} railways lines...", cfg)
        layer_style = LAYER_STYLES["railways"]
        thickness = layer_style["thickness"]
        color = grayscale_to_bgr(layer_style["color"])

        railways_start = time.time()
        drawn_count, draw_time = draw_lines_opencv(
            img,
            railways_lines,
            bbox_cached,
            cfg.width,
            cfg.height,
            thickness,
            color,
        )
        log_timing(railways_start, f"Railways rendering ({drawn_count} lines)", cfg)

    elif layer_name == "borders" and border_polygons:
        log(f"[Render] Drawing {len(border_polygons)} border polygons...", cfg)
        layer_style = LAYER_STYLES["borders"]
        color = grayscale_to_bgr(layer_style["color"])

        borders_start = time.time()
        drawn_count, draw_time = draw_polygons_opencv(
            img, border_polygons, bbox_cached, cfg.width, cfg.height, color
        )
        log_timing(borders_start, f"Borders rendering ({drawn_count} polygons)", cfg)

    # Generate output filename for this layer - determine final extension
    file_ext = cfg.output.suffix if cfg.output.suffix else ".exr"
    final_output_path = cfg.output.parent / f"{cfg.output.stem}_{layer_name}{file_ext}"

    # Create output directory
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    save_start = time.time()

    if file_ext.lower() == ".webp":
        # Write directly as WebP
        log(
            f"[Render] Saving {layer_name} layer as WebP to {final_output_path}...", cfg
        )

        # Convert float32 to 8-bit for WebP
        img_8bit = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
        webp_params = [cv2.IMWRITE_WEBP_QUALITY, 90]
        success = cv2.imwrite(str(final_output_path), img_8bit, webp_params)

        if not success:
            raise RuntimeError(f"Failed to save WebP image: {final_output_path}")
        log_timing(save_start, f"{layer_name} WebP file write", cfg)

        # Clean up
        del img
        del img_8bit

        return final_output_path

    elif file_ext.lower() == ".exr":
        # For EXR output, save to PNG first, then use oiiotool to convert
        log(
            f"[Render] Saving {layer_name} layer as PNG first, then converting to EXR...",
            cfg,
        )

        # Create temporary PNG path
        temp_png_path = cfg.output.parent / f"{cfg.output.stem}_{layer_name}.png"

        # Save as PNG first
        success = cv2.imwrite(str(temp_png_path), img)

        if not success:
            raise RuntimeError(f"Failed to save temporary PNG image: {temp_png_path}")
        log_timing(save_start, f"{layer_name} PNG file write", cfg)

        # Use oiiotool to convert PNG to EXR
        log("[Render] Converting PNG to EXR using oiiotool...", cfg)
        oiio_start = time.time()

        oiiotool_cmd = [
            "oiiotool",
            str(temp_png_path),
            "--tile",
            "64",
            "64",
            "-d",
            "half",
            "-compression",
            "zip",
            "-o",
            str(final_output_path),
        ]

        try:
            subprocess.run(oiiotool_cmd, check=True, capture_output=True, text=True)
            log_timing(oiio_start, f"{layer_name} oiiotool conversion", cfg)

            # Remove the temporary PNG file
            temp_png_path.unlink()
            log(f"[Render] Removed temporary PNG file: {temp_png_path}", cfg)

        except subprocess.CalledProcessError as e:
            # Clean up temp file even if conversion fails
            if temp_png_path.exists():
                temp_png_path.unlink()
            raise RuntimeError(f"oiiotool conversion failed: {e.stderr}")
        except FileNotFoundError:
            # Clean up temp file if oiiotool is not found
            if temp_png_path.exists():
                temp_png_path.unlink()
            raise RuntimeError(
                "oiiotool command not found. Please ensure OpenImageIO is installed."
            )

        # Clean up
        del img

        return final_output_path

    else:
        # For PNG output only
        png_output_path = cfg.output.parent / f"{cfg.output.stem}_{layer_name}.png"
        log(f"[Render] Saving {layer_name} layer as PNG to {png_output_path}...", cfg)

        success = cv2.imwrite(str(png_output_path), img)

        if not success:
            raise RuntimeError(f"Failed to save PNG image: {png_output_path}")
        log_timing(save_start, f"{layer_name} 8-bit PNG file write", cfg)

        # Clean up
        del img

        # For PNG output, rename the file to the final path
        png_output_path.rename(final_output_path)
        return final_output_path


def render(cfg, highway_layers, border_polygons, water_lines, railways_lines):
    """Render all requested layers to separate files"""
    output_paths = []

    for layer_name in cfg.render_types:
        if layer_name == "roads" and any(highway_layers.values()):
            path = render_single_layer(
                cfg,
                layer_name,
                highway_layers,
                border_polygons,
                water_lines,
                railways_lines,
            )
            output_paths.append(path)
        elif layer_name == "water" and water_lines:
            path = render_single_layer(
                cfg,
                layer_name,
                highway_layers,
                border_polygons,
                water_lines,
                railways_lines,
            )
            output_paths.append(path)
        elif layer_name == "railways" and railways_lines:
            path = render_single_layer(
                cfg,
                layer_name,
                highway_layers,
                border_polygons,
                water_lines,
                railways_lines,
            )
            output_paths.append(path)
        elif layer_name == "borders" and border_polygons:
            path = render_single_layer(
                cfg,
                layer_name,
                highway_layers,
                border_polygons,
                water_lines,
                railways_lines,
            )
            output_paths.append(path)

    return output_paths


# ---------------------- Main Flow --------------------------------------- #


def run(cfg):
    total_start = time.time()
    log(f"BBox: {cfg.bbox}", cfg, force=True)
    log(f"Output: {cfg.output}", cfg)
    log(f"Rendering: {', '.join(cfg.render_types)}", cfg, force=True)

    # Initialize data containers
    highway_layers = {tag: [] for tag in HIGHWAY_TAGS}
    border_polygons = []
    water_lines = []
    railways_lines = []

    # Highways
    if "roads" in cfg.render_types:
        highway_query = make_highways_query(cfg.bbox)
        highway_results = fetch_from_postgis(highway_query, cfg)

        extract_start = time.time()
        highway_layers = extract_highway_lines(highway_results)
        log_timing(extract_start, "Highway data extraction", cfg)

        totals = {k: len(v) for k, v in highway_layers.items()}
        log(f"Highway line counts: {totals}", cfg, force=True)

    # Borders
    if "borders" in cfg.render_types:
        border_query = make_borders_query(cfg.bbox)
        border_results = fetch_from_postgis(border_query, cfg)

        extract_start = time.time()
        border_polygons = extract_border_polygons(border_results)
        log_timing(extract_start, "Border data extraction", cfg)
        log(f"Border polygons: {len(border_polygons)}", cfg, force=True)

    # Water
    if "water" in cfg.render_types:
        water_query = make_water_query(cfg.bbox)
        water_results = fetch_from_postgis(water_query, cfg)

        extract_start = time.time()
        water_lines = extract_water_lines(water_results)
        log_timing(extract_start, "Water data extraction", cfg)
        log(f"Water way lines: {len(water_lines)}", cfg, force=True)

    # Railway
    if "railways" in cfg.render_types:
        railways_query = make_railways_query(cfg.bbox)
        railways_results = fetch_from_postgis(railways_query, cfg)

        extract_start = time.time()
        railways_lines = extract_railway_lines(railways_results)
        log_timing(extract_start, "Railways data extraction", cfg)
        log(f"Railways lines: {len(railways_lines)}", cfg, force=True)

    render_start = time.time()
    out_paths = render(
        cfg, highway_layers, border_polygons, water_lines, railways_lines
    )
    log_timing(render_start, "Total rendering", cfg)

    log_timing(total_start, "TOTAL EXECUTION TIME", cfg)

    if out_paths:
        log(
            f"Rendered {len(out_paths)} layer(s) (size {cfg.width}x{cfg.height}):",
            cfg,
            force=True,
        )
        for path in out_paths:
            log(f"  - {path}", cfg, force=True)
    else:
        log("No layers were rendered (no data found)", cfg, force=True)

    return out_paths


# ---------------------- CLI --------------------------------------------- #


def build_arg_parser():
    epilog = (
        "Notes: If your --bbox value starts with a negative number you must either quote it "
        "('--bbox \"-123.3,49.1,-122.9,49.35\"') or use the equals form (--bbox=-123.3,49.1,-122.9,49.35).\n"
        "Alternatively supply the four numeric components with --min-lon --min-lat --max-lon --max-lat.\n\n"
        "Render types: roads, borders, water, railways (all are optional, default is roads and borders)"
    )
    p = argparse.ArgumentParser(
        description="Render OSM highways + borders to various formats from PostGIS (pure Python)",
        epilog=epilog,
    )
    p.add_argument(
        "--bbox",
        required=False,
        type=str,
        default="130.36707314,32.49348870,130.85704370,32.98568071",
        help="Combined bbox 'min_lon,min_lat,max_lon,max_lat' (use equals sign for negative values: --bbox=-123.30,49.10,-122.90,49.35)",
    )
    # Component fallback arguments (optional if --bbox provided)
    p.add_argument("--min-lon", type=float, help="Minimum longitude (west)")
    p.add_argument("--min-lat", type=float, help="Minimum latitude (south)")
    p.add_argument("--max-lon", type=float, help="Maximum longitude (east)")
    p.add_argument("--max-lat", type=float, help="Maximum latitude (north)")
    p.add_argument(
        "--output",
        required=False,
        default="output/kumamoto.exr",
        help="Output file path (supports .exr, .png, .webp)",
    )
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--height", type=int, default=2048)
    p.add_argument(
        "--render",
        nargs="*",
        choices=["roads", "borders", "water", "railways"],
        default=["roads", "borders", "water", "railways"],
        help="What to render (default: roads borders). Can specify multiple: --render roads water borders",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Determine bbox source
    bbox = None
    if args.bbox:
        try:
            bbox = parse_bbox(args.bbox)
        except ValueError as e:  # noqa: BLE001
            parser.error(str(e))
    else:
        components = [args.min_lon, args.min_lat, args.max_lon, args.max_lat]
        if all(v is not None for v in components):
            try:
                bbox = (
                    float(args.min_lon),
                    float(args.min_lat),
                    float(args.max_lon),
                    float(args.max_lat),
                )
            except Exception as e:  # noqa: BLE001
                parser.error(f"Invalid numeric bbox components: {e}")
        else:
            parser.error(
                "You must supply either --bbox or all four of --min-lon --min-lat --max-lon --max-lat"
            )

    cfg = Config(
        bbox=bbox,
        width=args.width,
        height=args.height,
        output=Path(args.output),
        render_types=args.render,
        verbose=args.verbose,
    )

    try:
        run(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
