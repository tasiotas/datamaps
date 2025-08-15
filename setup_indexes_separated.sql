-- Optimized indexes for separated OSM tables
-- Run this after importing data with the separated Lua script

-- ============================================================================
-- ROADS TABLE INDEXES
-- ============================================================================
-- Note: osm_roads table only has 'highway' and 'way' columns (no osm_id)

-- Primary spatial index for roads
CREATE INDEX CONCURRENTLY idx_osm_roads_way 
ON osm_roads USING GIST (way);

-- Highway type index (very selective since all rows have highway values)
CREATE INDEX CONCURRENTLY idx_osm_roads_highway 
ON osm_roads (highway);

-- ============================================================================
-- WATERWAYS TABLE INDEXES
-- ============================================================================

-- Primary spatial index
CREATE INDEX CONCURRENTLY idx_osm_waterways_way 
ON osm_waterways USING GIST (way);

-- Waterway type index
CREATE INDEX CONCURRENTLY idx_osm_waterways_waterway 
ON osm_waterways (waterway);

-- Natural type index
CREATE INDEX CONCURRENTLY idx_osm_waterways_natural 
ON osm_waterways (natural);

-- Composite indexes for common queries
-- Note: Removed mixed GIST indexes (geometry + text) as they require explicit operator classes
-- Use separate B-tree indexes for text columns and GIST for spatial queries

-- OSM ID index
CREATE INDEX CONCURRENTLY idx_osm_waterways_osm_id 
ON osm_waterways (osm_id);

-- ============================================================================
-- RAILWAYS TABLE INDEXES
-- ============================================================================

-- Primary spatial index
CREATE INDEX CONCURRENTLY idx_osm_railways_way 
ON osm_railways USING GIST (way);

-- Railway type index
CREATE INDEX CONCURRENTLY idx_osm_railways_railway 
ON osm_railways (railway);

-- OSM ID index
CREATE INDEX CONCURRENTLY idx_osm_railways_osm_id 
ON osm_railways (osm_id);

-- ============================================================================
-- NATURAL FEATURES TABLE INDEXES
-- ============================================================================

-- Primary spatial index
CREATE INDEX CONCURRENTLY idx_osm_natural_way 
ON osm_natural USING GIST (way);

-- Natural type index
CREATE INDEX CONCURRENTLY idx_osm_natural_natural 
ON osm_natural (natural);

-- OSM ID index
CREATE INDEX CONCURRENTLY idx_osm_natural_osm_id 
ON osm_natural (osm_id);

-- ============================================================================
-- BOUNDARIES TABLE INDEXES
-- ============================================================================

-- Primary spatial index
CREATE INDEX CONCURRENTLY idx_osm_boundaries_way 
ON osm_boundaries USING GIST (way);

-- Boundary type index
CREATE INDEX CONCURRENTLY idx_osm_boundaries_boundary 
ON osm_boundaries (boundary);

-- Admin level index
CREATE INDEX CONCURRENTLY idx_osm_boundaries_admin_level 
ON osm_boundaries (admin_level);

-- Composite indexes
-- Note: Removed mixed GIST indexes (geometry + text) as they require explicit operator classes
-- Use separate B-tree indexes for text columns and GIST for spatial queries

CREATE INDEX CONCURRENTLY idx_osm_boundaries_boundary_admin_level 
ON osm_boundaries (boundary, admin_level);

-- OSM ID index
CREATE INDEX CONCURRENTLY idx_osm_boundaries_osm_id 
ON osm_boundaries (osm_id);

-- ISO country code index
CREATE INDEX CONCURRENTLY idx_osm_boundaries_iso_3166_1 
ON osm_boundaries (iso_3166_1) 
WHERE iso_3166_1 IS NOT NULL;

-- ============================================================================
-- TABLE STATISTICS AND MAINTENANCE
-- ============================================================================

-- Update table statistics for optimal query planning
ANALYZE osm_roads;
ANALYZE osm_waterways;
ANALYZE osm_railways;
ANALYZE osm_natural;
ANALYZE osm_boundaries;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT ON osm_roads TO your_user;
-- GRANT SELECT ON osm_waterways TO your_user;
-- GRANT SELECT ON osm_railways TO your_user;
-- GRANT SELECT ON osm_natural TO your_user;
-- GRANT SELECT ON osm_boundaries TO your_user;

-- ============================================================================
-- PERFORMANCE MONITORING QUERIES
-- ============================================================================

-- Check table sizes
/*
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename LIKE 'osm_%'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
*/

-- Check index usage
/*
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    most_common_vals
FROM pg_stats 
WHERE schemaname = 'public' 
    AND tablename LIKE 'osm_%'
ORDER BY tablename, attname;
*/
