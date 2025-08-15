-- Performance comparison between single table vs separated tables approach

-- ============================================================================
-- BEFORE: Single table query (your current approach)
-- ============================================================================

-- Your current query pattern
/*
EXPLAIN (ANALYZE, BUFFERS) 
SELECT
    highway,
    ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
FROM
    "public"."osm_filtered_data"
WHERE
    way && ST_Transform(ST_MakeEnvelope(128.46149013, 28.80136586, 146.98199382, 45.86018943, 4326), 3857)
    AND "highway" IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary');
*/

-- ============================================================================
-- AFTER: Separated tables query (new optimized approach)
-- ============================================================================

-- Much faster equivalent query using roads table
/*
EXPLAIN (ANALYZE, BUFFERS) 
SELECT
    highway,
    ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
FROM
    "public"."osm_roads"
WHERE
    way && ST_Transform(ST_MakeEnvelope(128.46149013, 28.80136586, 146.98199382, 45.86018943, 4326), 3857)
    AND highway IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary');
*/

-- ============================================================================
-- ADDITIONAL OPTIMIZED QUERIES FOR OTHER FEATURE TYPES
-- ============================================================================

-- Query waterways (rivers and coastlines)
/*
SELECT
    COALESCE(waterway, natural) as feature_type,
    name,
    ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
FROM
    osm_waterways
WHERE
    way && ST_Transform(ST_MakeEnvelope(128.46149013, 28.80136586, 146.98199382, 45.86018943, 4326), 3857);
*/

-- Query railways
/*
SELECT
    railway,
    name,
    ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
FROM
    osm_railways
WHERE
    way && ST_Transform(ST_MakeEnvelope(128.46149013, 28.80136586, 146.98199382, 45.86018943, 4326), 3857);
*/

-- Query natural water features
/*
SELECT
    natural,
    name,
    ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
FROM
    osm_natural
WHERE
    way && ST_Transform(ST_MakeEnvelope(128.46149013, 28.80136586, 146.98199382, 45.86018943, 4326), 3857)
    AND natural = 'water';
*/

-- Query administrative boundaries
/*
SELECT
    boundary,
    admin_level,
    name,
    iso_3166_1,
    ST_AsGeoJSON(ST_Transform(way, 4326)) AS geom
FROM
    osm_boundaries
WHERE
    way && ST_Transform(ST_MakeEnvelope(128.46149013, 28.80136586, 146.98199382, 45.86018943, 4326), 3857)
    AND boundary = 'administrative'
    AND admin_level = '2';
*/

-- ============================================================================
-- EXPECTED PERFORMANCE IMPROVEMENTS
-- ============================================================================

/*
PERFORMANCE COMPARISON:

BEFORE (Single Table):
- Table size: ~1M rows
- Relevant rows: ~454K highway rows  
- Query time: ~2.24s
- Index efficiency: Poor (scans many irrelevant rows)

AFTER (Separated Tables):
- Roads table: ~454K rows (only highways)
- Query time: Expected ~0.2-0.5s (80-90% improvement)
- Index efficiency: Excellent (only relevant data)

BENEFITS:
1. Smaller table size = faster scans
2. More selective indexes
3. Better query planner statistics
4. Easier maintenance and optimization
5. Cleaner separation of concerns
6. Parallel queries across different feature types
*/

-- ============================================================================
-- MIGRATION STRATEGY
-- ============================================================================

/*
TO MIGRATE FROM CURRENT SETUP:

1. Test the new approach:
   - Use ingest_separated.lua to import into new tables
   - Run setup_indexes_separated.sql to create indexes
   - Test queries and measure performance

2. Update your application:
   - Modify render_osm.py to query osm_roads instead of osm_filtered_data
   - Update QGIS layers to point to new tables
   - Adjust any other scripts/tools

3. Drop old table (after confirming everything works):
   - DROP TABLE osm_filtered_data;

4. Monitor and optimize:
   - Use EXPLAIN ANALYZE to verify performance
   - Adjust indexes as needed based on actual usage patterns
*/
