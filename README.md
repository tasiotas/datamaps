# OSM Highway Renderer (Pure Python)

A compact Python script (`render_osm.py`) that fetches OpenStreetMap highways (motorway, trunk, primary, secondary, tertiary) and national borders (admin_level=2) for a bounding box and renders them into a PNG image using supersampling for smooth lines.

## Install

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python render_osm.py \
  --bbox -123.30,49.10,-122.90,49.35 \
  --output output/vancouver_test.png \
  --width 2048 --height 1536 \
  --scale 4 --verbose
```

### Arguments

* `--bbox` (required) `min_lon,min_lat,max_lon,max_lat`
* `--output` (required) Output PNG path
* `--width` / `--height` Image dimensions in final pixels (default 2048x1536)
* `--scale` Supersampling scale factor (default 4) for antialiasing
* `--retries` Overpass retry attempts (default 3)
* `--timeout` Per-request timeout seconds (default 180)
* `--dump-dir` Directory to write per-layer GeoJSON (optional)
* `--no-borders` Skip border query / rendering
* `--verbose` More logging

## Output

The resulting image is white background with black lines of varying thickness:
(tertiary < secondary < primary < trunk < motorway < borders).

## Notes

* Be considerate of Overpass API usage; large bboxes may time out.
* The script performs simple relation member handling for borders by drawing each way independently.
* You can post-process the PNG (e.g., invert colors, recolor lines) with standard image tools if needed.

## License

MIT (adjust as desired).
