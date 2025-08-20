-- Optimized OSM2PGSQL Lua filter that creates separate tables for different feature types.
-- This approach provides much better query performance by avoiding large table scans.

-- Define separate tables for each feature type
local roads_table = osm2pgsql.define_table {
    name = 'osm_roads',
    columns = {
        { column = 'highway', type = 'text', not_null = true },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

local railways_table = osm2pgsql.define_table {
    name = 'osm_railways',
    columns = {
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

local natural_features_table = osm2pgsql.define_table {
    name = 'osm_water',
    columns = {
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

local boundaries_table = osm2pgsql.define_table {
    name = 'osm_boundaries',
    columns = {
        { column = 'iso_3166_1', type = 'text' },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

-- Helper functions to insert into appropriate tables
local function insert_road(object, highway_type)
    local row = {
        highway = highway_type,
        way = object:as_linestring()
    }
    roads_table:insert(row)
end

local function insert_railway(object)
    local row = {
        way = object:as_linestring()
    }
    
    railways_table:insert(row)
end

local function insert_water(object, natural_type)
    local row = {
    }
    
    -- Handle geometry based on object type and feature type
    if object.tags and object.tags.type == 'multipolygon' then
        row.way = object:as_multipolygon()
    elseif natural_type == 'water' and object.is_closed then
        row.way = object:as_polygon()
    else
        -- For ways, use linestring; for relations, use multipolygon
        local osm_type = object.object_type or 'way'
        if osm_type == 'way' then
            row.way = object:as_linestring()
        else
            row.way = object:as_multipolygon()
        end
    end
    
    natural_features_table:insert(row)
end

local function insert_boundary(object, boundary_type, admin_level, object_type)
    local row = {
        iso_3166_1 = object.tags['ISO3166-1'],
        way = object:as_multipolygon()
    }
        
    boundaries_table:insert(row)
end

-- Process nodes: We skip all of them.
function osm2pgsql.process_node(object)
    return nil
end

-- Process ways
function osm2pgsql.process_way(object)
    local tags = object.tags
    
    -- Roads/Highways
    if tags.highway and (
        tags.highway == 'motorway' or
        tags.highway == 'trunk' or
        tags.highway == 'primary' or
        tags.highway == 'secondary' or
        tags.highway == 'tertiary'
    ) then
        insert_road(object, tags.highway)
        return
    end

    -- Railways
    if tags.railway == 'rail' then
        insert_railway(object)
        return
    end

    -- Waterways and water features
    if tags.waterway == 'river' or tags.natural == 'water' then
        insert_water(object, tags.natural)
        return
    end

end

-- Process relations
function osm2pgsql.process_relation(object)
    local tags = object.tags
    
    -- Only process multipolygon or boundary relations
    if not (tags.type == 'multipolygon' or tags.type == 'boundary') then
        return
    end
    
    -- Railways
    if tags.railway == 'rail' then
        insert_railway(object)
        return
    end

    -- Waterways and water features
    if tags.waterway == 'river' or tags.natural == 'water' then
        insert_water(object, tags.natural)
        return
    end

    -- Administrative boundaries
    if tags.boundary == 'administrative' and tags.admin_level == '2' then
        insert_boundary(object, tags.boundary, tags.admin_level, 'relation')
        return
    end
end
