# Define output resolution
FINAL_X=14000
FINAL_Y=14000

# Define a scaling factor for anti-aliasing (e.g., 4x)
SCALE=4
SUPER_X=$(($FINAL_X * $SCALE))
SUPER_Y=$(($FINAL_Y * $SCALE))

gdal_rasterize \
    -ot Byte \
    -a_nodata 0 \
    -burn 255 \
    -ts $SUPER_X $SUPER_Y \
    -co "COMPRESS=LZW" \
    -sql "SELECT ST_Buffer(way, \
              CASE highway \
                WHEN 'motorway' THEN 16.0 \
                WHEN 'trunk'    THEN 8.0 \
                WHEN 'primary'  THEN 4.0 \
                WHEN 'secondary'THEN 2.0 \
                WHEN 'tertiary' THEN 1.0 \
                ELSE 5.0 \
              END * $SCALE \
          ) FROM public.osm_roads \
          WHERE way && ST_Transform(ST_MakeEnvelope(139.50, 35.40, 140.05, 35.90, 4326), 3857) \
          AND highway IN ('tertiary', 'primary', 'secondary', 'trunk', 'motorway')" \
    PG:"host=192.168.1.3 port=5432 user=postgres dbname=osm_db" \
    tokyo_roads_supersampled.tif

oiiotool \
  tokyo_roads_supersampled.tif \
  --resize:filter=lanczos3 25% \
  -o tokyo_roads_final.png