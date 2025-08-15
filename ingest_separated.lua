-- Optimized OSM2PGSQL Lua filter that creates separate tables for different feature types.
-- This approach provides much better query performance by avoiding large table scans.

-- Define separate tables for each feature type
local roads_table = osm2pgsql.define_table {
    name = 'osm_roads',
    columns = {
        -- { column = 'osm_id', type = 'int8' },
        -- { column = 'osm_type', type = 'text' },
        { column = 'highway', type = 'text', not_null = true },
        -- { column = 'name', type = 'text' },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

local waterways_table = osm2pgsql.define_table {
    name = 'osm_waterways',
    columns = {
        { column = 'osm_id', type = 'int8' },
        { column = 'osm_type', type = 'text' },
        { column = 'waterway', type = 'text' },
        { column = 'natural', type = 'text' },
        { column = 'name', type = 'text' },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

local railways_table = osm2pgsql.define_table {
    name = 'osm_railways',
    columns = {
        -- { column = 'osm_id', type = 'int8' },
        -- { column = 'osm_type', type = 'text' },
        { column = 'railway', type = 'text', not_null = true },
        -- { column = 'name', type = 'text' },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

local natural_features_table = osm2pgsql.define_table {
    name = 'osm_natural',
    columns = {
        { column = 'osm_id', type = 'int8' },
        { column = 'osm_type', type = 'text' },
        { column = 'natural', type = 'text', not_null = true },
        { column = 'name', type = 'text' },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

local boundaries_table = osm2pgsql.define_table {
    name = 'osm_boundaries',
    columns = {
        { column = 'osm_id', type = 'int8' },
        { column = 'osm_type', type = 'text' },
        { column = 'boundary', type = 'text', not_null = true },
        { column = 'admin_level', type = 'text' },
        { column = 'name', type = 'text' },
        { column = 'iso_3166_1', type = 'text' },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

-- Helper functions to insert into appropriate tables
local function insert_road(object, highway_type)
    local row = {
        -- osm_id = object.id,
        -- osm_type = (object.tags and 'way') or 'relation',
        highway = highway_type,
        -- name = object.tags.name,
        way = object:as_linestring()
    }
    roads_table:insert(row)
end

local function insert_waterway(object, waterway_type, natural_type)
    local row = {
        osm_id = object.id,
        osm_type = object.object_type or 'way',  -- Use object.object_type for reliable type detection
        waterway = waterway_type,
        natural = natural_type,
        name = object.tags.name
    }
    
    -- Handle geometry based on object type and feature type
    if object.tags and object.tags.type == 'multipolygon' then
        row.way = object:as_multipolygon()
    elseif natural_type == 'water' and object.is_closed then
        row.way = object:as_polygon()
    else
        -- For ways, use linestring; for relations, use multipolygon
        if row.osm_type == 'way' then
            row.way = object:as_linestring()
        else
            row.way = object:as_multipolygon()
        end
    end
    
    waterways_table:insert(row)
end

local function insert_railway(object, railway_type)
    local row = {
        -- osm_id = object.id,
        -- osm_type = object.object_type or 'way',  -- Use object.object_type for reliable type detection
        railway = railway_type,
        name = object.tags.name,
        way = object:as_linestring()
    }
    
    -- -- Handle geometry based on object type
    -- if row.osm_type == 'way' then
    --     row.way = object:as_linestring()
    -- else
    --     row.way = object:as_multipolygon()
    -- end
    
    railways_table:insert(row)
end

local function insert_natural(object, natural_type)
    local row = {
        osm_id = object.id,
        osm_type = object.object_type or 'way',  -- Use object.object_type for reliable type detection
        natural = natural_type,
        name = object.tags.name
    }
    
    -- Handle geometry based on object type
    if object.tags and object.tags.type == 'multipolygon' then
        row.way = object:as_multipolygon()
    elseif natural_type == 'water' and object.is_closed then
        row.way = object:as_polygon()
    else
        -- For ways, use linestring; for relations without multipolygon type, try multipolygon
        if row.osm_type == 'way' then
            row.way = object:as_linestring()
        else
            row.way = object:as_multipolygon()
        end
    end
    
    natural_features_table:insert(row)
end

local function insert_boundary(object, boundary_type, admin_level, object_type)
    local row = {
        osm_id = object.id,
        osm_type = object_type,
        boundary = boundary_type,
        admin_level = admin_level,
        name = object.tags.name,
        iso_3166_1 = object.tags['ISO3166-1']
    }
    
    -- Handle geometry based on object type
    if object.tags and object.tags.type == 'multipolygon' then
        row.way = object:as_multipolygon()
    else
        -- For ways, use linestring; for relations, use multipolygon
        if object_type == 'way' then
            row.way = object:as_linestring()
        else
            row.way = object:as_multipolygon()
        end
    end
    
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

    -- Waterways and water features
    if tags.waterway == 'river' then
        insert_waterway(object, tags.waterway, nil)
        return
    end
    
    if tags.natural == 'coastline' then
        insert_waterway(object, nil, tags.natural)
        return
    end

    -- Railways
    if tags.railway == 'rail' then
        insert_railway(object, tags.railway)
        return
    end

    -- Natural water bodies
    if tags.natural == 'water' then
        insert_natural(object, tags.natural)
        return
    end

    -- Administrative boundaries
    if tags.boundary == 'administrative' and tags.admin_level == '2' then
        insert_boundary(object, tags.boundary, tags.admin_level, 'way')
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
    
    -- -- Roads/Highways (rare but possible in relations)
    -- if tags.highway and (
    --     tags.highway == 'motorway' or
    --     tags.highway == 'trunk' or
    --     tags.highway == 'primary' or
    --     tags.highway == 'secondary'
    -- ) then
    --     insert_road(object, tags.highway)
    --     return
    -- end

    -- Waterways and water features
    if tags.waterway == 'river' then
        insert_waterway(object, tags.waterway, nil)
        return
    end
    
    if tags.natural == 'coastline' then
        insert_waterway(object, nil, tags.natural)
        return
    end

    -- Railways
    if tags.railway == 'rail' then
        insert_railway(object, tags.railway)
        return
    end

    -- Natural water bodies
    if tags.natural == 'water' then
        insert_natural(object, tags.natural)
        return
    end

    -- Administrative boundaries
    if tags.boundary == 'administrative' and tags.admin_level == '2' then
        insert_boundary(object, tags.boundary, tags.admin_level, 'relation')
        return
    end
end
