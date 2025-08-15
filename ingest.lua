-- Simple OSM2PGSQL Lua filter that defines and populates a single table.

-- Function to check if a feature's tags should be kept.
local function get_filtered_tags(tags)
    -- Keep highways
    if tags.highway and (
        tags.highway == 'motorway' or
        tags.highway == 'trunk' or
        tags.highway == 'primary' or
        tags.highway == 'secondary' or
        tags.highway == 'tertiary'
    ) then
        return tags
    end

    -- Keep waterways
    if tags.waterway == 'river' or tags.natural == 'coastline' then
        return tags
    end

    -- Keep railways
    if tags.railway == 'rail' then
        return tags
    end

    -- Keep natural water bodies
    if tags.natural == 'water' then
        return tags
    end

    -- Keep administrative boundaries
    if tags.boundary == 'administrative' and tags.admin_level == '2' then
        return tags
    end

    return nil
end

-- Define the output table once.
-- We are creating a single table named 'osm_filtered_data' to hold all our features.
local output_table = osm2pgsql.define_table {
    name = 'osm_filtered_data',
    columns = {
        { column = 'osm_id', type = 'int8' },
        { column = 'osm_type', type = 'text' }, -- Add a column to track original type
        { column = 'highway', type = 'text', tags = { highway = true } },
        { column = 'waterway', type = 'text', tags = { waterway = true } },
        { column = 'railway', type = 'text', tags = { railway = true } },
        { column = 'natural', type = 'text', tags = { natural = true } },
        { column = 'boundary', type = 'text', tags = { boundary = true } },
        { column = 'name', type = 'text', tags = { name = true } },
		{ column = 'iso_3166_1', type = 'text', tags = { ['ISO3166-1'] = true } },
        { column = 'way', type = 'geometry', projection = 3857, not_null = true }
    }
}

-- Process nodes: We skip all of them.
function osm2pgsql.process_node(object)
    return nil
end

-- Process ways
function osm2pgsql.process_way(object)
    local tags_to_keep = get_filtered_tags(object.tags)

    if tags_to_keep then
        local row = {}

        row.osm_id = object.id
        row.osm_type = 'way'
        row.highway = tags_to_keep.highway
        row.waterway = tags_to_keep.waterway
        row.railway = tags_to_keep.railway
        row.natural = tags_to_keep.natural
        row.boundary = tags_to_keep.boundary
        row.name = object.tags.name

        if (tags_to_keep.natural == 'water') and object.is_closed then
            row.way = object:as_polygon()
        else
            row.way = object:as_linestring()
        end

        output_table:insert(row)
    end
end

-- Process relations
function osm2pgsql.process_relation(object)
    local tags_to_keep = get_filtered_tags(object.tags)

    -- Check if it's a multipolygon or a boundary relation and if we want to keep its tags
    if tags_to_keep and (object.tags.type == 'multipolygon' or object.tags.type == 'boundary') then
        local row = {}
        row.osm_id = object.id
        row.osm_type = 'relation'

        -- Copy the relevant tags
        row.highway = tags_to_keep.highway
        row.waterway = tags_to_keep.waterway
        row.railway = tags_to_keep.railway
        row.natural = tags_to_keep.natural
        row.boundary = tags_to_keep.boundary
        row.name = object.tags.name
		row.iso_3166_1 = tags_to_keep['ISO3166-1']

        -- Convert the relation to a multipolygon geometry, as `osm2pgsql` will correctly handle `type=boundary`
        row.way = object:as_multipolygon()

        -- Insert the row into the output table
        output_table:insert(row)
    end
end