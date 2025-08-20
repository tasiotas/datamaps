-- ============================================================================
-- POST-INGEST OPTIMIZATION SCRIPT
-- Optimized for map rendering queries with spatial + attribute filtering
-- ============================================================================

-- ============================================================================
-- COUNTRY BOUNDARIES SETUP
-- ============================================================================
-- Create labeled countries by joining land polygons with OSM boundaries
CREATE TABLE IF NOT EXISTS labeled_countries AS
SELECT
    c.wkb_geometry AS way,
    l.iso_3166_1
FROM
    countries c
JOIN
    osm_boundaries l
ON
    ST_Intersects(c.wkb_geometry, l.way)
WHERE
    l.iso_3166_1 IS NOT NULL;

-- Create spatial index for labeled_countries
CREATE INDEX CONCURRENTLY idx_labeled_countries_way 
ON labeled_countries USING GIST (way);

-- Drop the original countries table
DROP TABLE IF EXISTS countries;

-- Drop osm_boundaries table (data now in labeled_countries)
DROP TABLE IF EXISTS osm_boundaries;



-- ROADS: Highway type index (essential for zoom-level filtering)
-- This dramatically improves queries like: WHERE highway IN ('motorway', 'trunk', 'primary')
CREATE INDEX CONCURRENTLY idx_osm_roads_highway 
ON osm_roads (highway);

-- ============================================================================
-- QUERY OPTIMIZATION SETTINGS
-- ============================================================================

-- Increase work_mem for this session to speed up index creation
SET work_mem = '256MB';

-- Update table statistics for optimal query planning
ANALYZE osm_roads;
ANALYZE osm_water;        
ANALYZE osm_railways;
ANALYZE labeled_countries;


-- Reset work_mem to default
RESET work_mem;
