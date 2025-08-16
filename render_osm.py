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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "borders": {"thickness": 6, "color": 1},  # Dark gray for borders
    #
    "motorway": {"thickness": 2, "color": 1},  # Very dark gray (main roads)
    "trunk": {"thickness": 1, "color": 1},  # Dark gray
    "primary": {"thickness": 1, "color": 0.8},  # Medium-dark gray
    "secondary": {"thickness": 1, "color": 0.6},  # Medium gray
    "tertiary": {"thickness": 1, "color": 0.2},  # Lighter gray (brighter)
    #
    "water": {"thickness": 1, "color": 1},  # Medium-dark gray for water
    "railways": {"thickness": 1, "color": 1},  # Dark gray for railways
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
        geojson_polygon=None,
    ):
        self.bbox = bbox  # (min_lon, min_lat, max_lon, max_lat)
        self.width = width
        self.height = height
        self.output = output if isinstance(output, Path) else Path(output)
        self.render_types = render_types or ["roads", "borders"]
        self.verbose = verbose
        self.geojson_polygon = (
            geojson_polygon  # GeoJSON polygon geometry for precise filtering
        )

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


def parse_geojson_region(geojson_path):
    """Parse GeoJSON file and extract bounding box and polygon geometry from the first polygon feature"""
    try:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to read or parse GeoJSON file '{geojson_path}': {e}")

    # Extract coordinates from the first feature
    features = geojson_data.get("features", [])
    if not features:
        raise ValueError("No features found in GeoJSON file")

    first_feature = features[0]
    geometry = first_feature.get("geometry", {})
    geom_type = geometry.get("type", "")
    coordinates = geometry.get("coordinates", [])

    if geom_type != "Polygon":
        raise ValueError(f"Expected Polygon geometry, found {geom_type}")

    if not coordinates or not coordinates[0]:
        raise ValueError("Invalid polygon coordinates in GeoJSON")

    # Extract exterior ring coordinates (first array in coordinates)
    exterior_ring = coordinates[0]

    # Calculate bounding box from all coordinates
    lons = [coord[0] for coord in exterior_ring]
    lats = [coord[1] for coord in exterior_ring]

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("Invalid polygon extents")

    # Return both bbox and the full geometry for precise spatial queries
    return (min_lon, min_lat, max_lon, max_lat), geometry


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


def fetch_data_parallel(cfg):
    """Fetch all required data types in parallel"""
    query_tasks = []

    # Prepare query tasks based on what needs to be rendered
    if "roads" in cfg.render_types:
        query_tasks.append(
            (
                "highways",
                make_highways_query(cfg.bbox, cfg.geojson_polygon),
                extract_highway_lines,
            )
        )

    if "borders" in cfg.render_types:
        query_tasks.append(
            (
                "borders",
                make_borders_query(cfg.bbox, cfg.geojson_polygon),
                extract_border_polygons,
            )
        )

    if "water" in cfg.render_types:
        query_tasks.append(
            (
                "water",
                make_water_query(cfg.bbox, cfg.geojson_polygon),
                extract_water_features,
            )
        )

    if "railways" in cfg.render_types:
        query_tasks.append(
            (
                "railways",
                make_railways_query(cfg.bbox, cfg.geojson_polygon),
                extract_railway_lines,
            )
        )

    log(
        f"[Data] Starting {len(query_tasks)} parallel database queries...",
        cfg,
        force=True,
    )

    # Execute queries in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all queries
        future_to_task = {}
        for task_name, query, extract_func in query_tasks:
            log(f"[Data] Submitting {task_name} query...", cfg)
            future = executor.submit(fetch_from_postgis, query, cfg)
            future_to_task[future] = (task_name, extract_func)

        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_task):
            task_name, extract_func = future_to_task[future]
            try:
                query_results = future.result()
                extract_start = time.time()
                extracted_data = extract_func(query_results)
                log_timing(extract_start, f"{task_name} data extraction", cfg)
                results[task_name] = extracted_data
                completed_count += 1

                # Log extracted data counts
                if task_name == "highways":
                    totals = {k: len(v) for k, v in extracted_data.items()}
                    log(f"Highway line counts: {totals}", cfg, force=True)
                elif task_name == "borders":
                    log(f"Border polygons: {len(extracted_data)}", cfg, force=True)
                elif task_name == "water":
                    lines, polygons = extracted_data
                    log(
                        f"Water lines: {len(lines)}, Water polygons: {len(polygons)}",
                        cfg,
                        force=True,
                    )
                elif task_name == "railways":
                    log(f"Railways lines: {len(extracted_data)}", cfg, force=True)

                log(
                    f"[Data] Completed {task_name} ({completed_count}/{len(query_tasks)})",
                    cfg,
                )

            except Exception as e:
                log(f"[ERROR] Failed to fetch {task_name}: {e}", cfg, force=True)
                results[task_name] = None

    return results


# ---------------------- Query Construction ------------------------------ #


def make_spatial_filter(bbox, geojson_polygon=None):
    """Create spatial filter clause - use polygon geometry if available, otherwise bbox"""
    if geojson_polygon:
        # Convert GeoJSON polygon to WKT format for PostGIS
        coords = geojson_polygon["coordinates"][0]  # exterior ring
        wkt_coords = ", ".join([f"{lon} {lat}" for lon, lat in coords])
        polygon_wkt = f"POLYGON(({wkt_coords}))"
        return f"ST_Intersects(way, ST_Transform(ST_GeomFromText('{polygon_wkt}', 4326), 3857))"
    else:
        # Fall back to bounding box
        min_lon, min_lat, max_lon, max_lat = bbox
        return f"way && ST_Transform(ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326), 3857)"


def make_borders_query(bbox, geojson_polygon=None):
    spatial_filter = make_spatial_filter(bbox, geojson_polygon)
    query = f"""
    SELECT
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."labeled_countries"
    WHERE
        {spatial_filter};
    """
    return query


def make_highways_query(bbox, geojson_polygon=None):
    highway_filter = ", ".join([f"'{tag}'" for tag in HIGHWAY_TAGS])
    spatial_filter = make_spatial_filter(bbox, geojson_polygon)
    query = f"""
    SELECT
        highway,
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."osm_roads"
    WHERE
        {spatial_filter}
        AND "highway" IN ({highway_filter});
    """
    return query


def make_water_query(bbox, geojson_polygon=None):
    spatial_filter = make_spatial_filter(bbox, geojson_polygon)
    query = f"""
    SELECT
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."osm_water"
    WHERE
        {spatial_filter}
    """
    return query


def make_railways_query(bbox, geojson_polygon=None):
    spatial_filter = make_spatial_filter(bbox, geojson_polygon)
    query = f"""
    SELECT
        ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
    FROM
        "public"."osm_railways"
    WHERE
        {spatial_filter}
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


def extract_water_features(results):
    """Extract water features, separating lines (rivers) from polygons (lakes)"""
    lines = []
    polygons = []

    for row in results:
        geom_json = json.loads(row["geom"])
        geom_type = geom_json.get("type", "")
        coords = geom_json.get("coordinates", [])

        if geom_type == "LineString":
            # LineString: coordinates = [[lon, lat], [lon, lat], ...]
            if len(coords) >= 2:
                lines.append(coords)
        elif geom_type == "MultiLineString":
            # MultiLineString: coordinates = [[[lon, lat], [lon, lat], ...], ...]
            for line_coords in coords:
                if len(line_coords) >= 2:
                    lines.append(line_coords)
        elif geom_type == "Polygon":
            # Polygon: coordinates = [[[lon, lat], [lon, lat], ...]] (exterior ring only)
            if coords and len(coords[0]) >= 3:
                polygons.append(coords[0])
        elif geom_type == "MultiPolygon":
            # MultiPolygon: coordinates = [[[[lon, lat], [lon, lat], ...]], ...]
            for polygon in coords:
                if polygon and len(polygon[0]) >= 3:
                    polygons.append(polygon[0])

    return lines, polygons


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

    # Ensure coords is a proper 2D array with consistent shape
    try:
        coords_array = np.array(coords)

        # Validate that we have at least 2D coordinates
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            # Try to handle inconsistent coordinate formats
            valid_coords = []
            for coord in coords:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    valid_coords.append([float(coord[0]), float(coord[1])])

            if not valid_coords:
                return np.array([], dtype=np.int32).reshape(0, 2)

            coords_array = np.array(valid_coords)
    except (ValueError, TypeError):
        # Handle malformed coordinate data
        return np.array([], dtype=np.int32).reshape(0, 2)

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

    for polygon in polygons:
        if len(polygon) < 3:  # Need at least 3 points for a polygon
            continue

        # Convert coordinates in batch using numpy with sub-pixel precision
        pts = lonlat_to_pixel_batch(polygon, bbox, scaled_w, scaled_h, shift)

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

        # Check if we have valid points to draw
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
    cfg,
    layer_name,
    highway_layers,
    border_polygons,
    water_lines,
    water_polygons,
    railways_lines,
):
    """Render a single layer type to its own image"""
    # Create OpenCV image (32-bit float format, black background) for better antialiasing visibility
    img = np.full((cfg.height, cfg.width, 3), 0, dtype=np.uint8)

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

    elif layer_name == "water" and (water_lines or water_polygons):
        layer_style = LAYER_STYLES["water"]
        thickness = layer_style["thickness"]
        color = grayscale_to_bgr(layer_style["color"])

        water_start = time.time()
        total_drawn = 0

        # Draw water polygons (lakes, ponds) as filled shapes
        if water_polygons:
            log(
                f"[Render] Drawing {len(water_polygons)} water polygons (filled)...",
                cfg,
            )
            drawn_polygons, _ = draw_polygons_opencv(
                img,
                water_polygons,
                bbox_cached,
                cfg.width,
                cfg.height,
                color,
            )
            total_drawn += drawn_polygons

        # Draw water lines (rivers, streams) as strokes
        if water_lines:
            log(f"[Render] Drawing {len(water_lines)} water lines...", cfg)
            drawn_lines, _ = draw_lines_opencv(
                img,
                water_lines,
                bbox_cached,
                cfg.width,
                cfg.height,
                thickness,
                color,
            )
            total_drawn += drawn_lines

        log_timing(water_start, f"Water rendering ({total_drawn} features)", cfg)

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


def render(
    cfg, highway_layers, border_polygons, water_lines, water_polygons, railways_lines
):
    """Render all requested layers to separate files in parallel"""

    # Prepare rendering tasks
    render_tasks = []

    for layer_name in cfg.render_types:
        if layer_name == "roads" and highway_layers and any(highway_layers.values()):
            render_tasks.append(
                (
                    layer_name,
                    highway_layers,
                    border_polygons,
                    water_lines,
                    water_polygons,
                    railways_lines,
                )
            )
        elif layer_name == "water" and (water_lines or water_polygons):
            render_tasks.append(
                (
                    layer_name,
                    highway_layers,
                    border_polygons,
                    water_lines,
                    water_polygons,
                    railways_lines,
                )
            )
        elif layer_name == "railways" and railways_lines:
            render_tasks.append(
                (
                    layer_name,
                    highway_layers,
                    border_polygons,
                    water_lines,
                    water_polygons,
                    railways_lines,
                )
            )
        elif layer_name == "borders" and border_polygons:
            render_tasks.append(
                (
                    layer_name,
                    highway_layers,
                    border_polygons,
                    water_lines,
                    water_polygons,
                    railways_lines,
                )
            )

    log(
        f"[Render] Starting {len(render_tasks)} parallel rendering tasks...",
        cfg,
        force=True,
    )

    # Render layers in parallel
    output_paths = []
    if render_tasks:
        with ThreadPoolExecutor(max_workers=min(len(render_tasks), 4)) as executor:
            # Submit all rendering tasks
            future_to_layer = {}
            for task_args in render_tasks:
                layer_name = task_args[0]
                log(f"[Render] Submitting {layer_name} rendering task...", cfg)
                future = executor.submit(render_single_layer, cfg, *task_args)
                future_to_layer[future] = layer_name

            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_layer):
                layer_name = future_to_layer[future]
                try:
                    output_path = future.result()
                    output_paths.append(output_path)
                    completed_count += 1
                    log(
                        f"[Render] Completed {layer_name} layer ({completed_count}/{len(render_tasks)}): {output_path}",
                        cfg,
                        force=True,
                    )
                except Exception as e:
                    log(f"[ERROR] Failed to render {layer_name}: {e}", cfg, force=True)

    return output_paths


# ---------------------- Main Flow --------------------------------------- #


def run(cfg):
    total_start = time.time()
    log(f"BBox: {cfg.bbox}", cfg, force=True)
    log(f"Output: {cfg.output}", cfg)
    log(f"Rendering: {', '.join(cfg.render_types)}", cfg, force=True)

    # Fetch all required data in parallel
    log("[Data] Starting parallel data fetch...", cfg)
    fetch_start = time.time()
    data_results = fetch_data_parallel(cfg)
    log_timing(fetch_start, "Total parallel data fetch", cfg)

    # Extract data from results with defaults
    highway_layers = data_results.get("highways", {tag: [] for tag in HIGHWAY_TAGS})
    border_polygons = data_results.get("borders", [])
    water_data = data_results.get("water", ([], []))
    water_lines, water_polygons = (
        water_data if isinstance(water_data, tuple) else (water_data, [])
    )
    railways_lines = data_results.get("railways", [])

    # Start parallel rendering
    log("[Render] Starting parallel rendering...", cfg)
    render_start = time.time()
    out_paths = render(
        cfg,
        highway_layers,
        border_polygons,
        water_lines,
        water_polygons,
        railways_lines,
    )
    log_timing(render_start, "Total parallel rendering", cfg)

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
        "You can also use --geojson to specify a polygon region from a GeoJSON file (overrides --bbox).\n\n"
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
    p.add_argument(
        "--geojson",
        required=False,
        type=str,
        help="Path to GeoJSON file containing a polygon region to render (overrides --bbox if provided)",
    )
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

    # Determine bbox source - prioritize GeoJSON if provided
    bbox = None
    geojson_polygon = None

    if args.geojson:
        try:
            bbox, geojson_polygon = parse_geojson_region(args.geojson)
            print(f"Using region from GeoJSON file: {args.geojson}", file=sys.stderr)
            print(f"Extracted bbox: {bbox}", file=sys.stderr)
            print(
                "Using precise polygon geometry for spatial filtering", file=sys.stderr
            )
        except ValueError as e:
            parser.error(str(e))
    else:
        # Fall back to bbox argument
        try:
            bbox = parse_bbox(args.bbox)
        except ValueError as e:  # noqa: BLE001
            parser.error(str(e))

    cfg = Config(
        bbox=bbox,
        width=args.width,
        height=args.height,
        output=Path(args.output),
        render_types=args.render,
        verbose=args.verbose,
        geojson_polygon=geojson_polygon,
    )

    try:
        run(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
