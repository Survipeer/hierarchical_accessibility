import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import json
import pickle
import os

def generate_hex_map(resolution, bounding_box):
    """
    Generate a hexagonal grid covering the specified area and return the centroids.
    """
    min_lat, min_lon, max_lat, max_lon = bounding_box
    
    # Generate hexagons using the center point method
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Get the base hexagon
    base_hex = h3.latlng_to_cell(center_lat, center_lon, resolution)
    
    # Define the 4 corners of the bounding box
    corners = [
        (min_lat, min_lon),
        (max_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon)
    ]
    print("polygon:", corners)
    # Dynamically find the required k (number of rings) to reach the furthest corner
    required_k = 0
    for lat, lon in corners:
        corner_hex = h3.latlng_to_cell(lat, lon, resolution)
        try:
            # grid_distance returns the number of rings between two cells
            dist = h3.grid_distance(base_hex, corner_hex)
            if dist > required_k:
                required_k = dist
        except h3.H3Error:
            # Fallback for extremely large bounding boxes crossing icosahedron edges
            required_k = max(required_k, 200) 
            
    # Add a buffer of 2 rings to ensure the extreme edges are comfortably covered
    k = required_k + 2
    
    # Generate the grid disk with our dynamic k
    hexagons = h3.grid_disk(base_hex, k)
    
    # Filter hexagons to only include those within the bounding box
    filtered_hexagons = []
    for hex_id in hexagons:
        hex_lat, hex_lon = h3.cell_to_latlng(hex_id)
        if (min_lat <= hex_lat <= max_lat) and (min_lon <= hex_lon <= max_lon):
            filtered_hexagons.append({
                'hex_id': hex_id,
                'lat': hex_lat,
                'lon': hex_lon
            })
    print("Total Hexes:", len(filtered_hexagons))
    return filtered_hexagons

def export_to_csv(centroids, NETWORK_NAME, HUB_COUNT):
    """
    Export the hex centroids to a CSV file.
    
    Args:
        centroids: List of dictionaries with hex_id, lat, lon
        filename: Output filename
    """
    df = pd.DataFrame(centroids)
    df.to_csv(f'./Data/hex_hash_map/{NETWORK_NAME}_{HUB_COUNT}/{NETWORK_NAME}_{HUB_COUNT}_hex_centroid.csv', index=False)
    print(f"Exported {len(centroids)} hex centroids")

def export_to_geojson(centroids, NETWORK_NAME, HUB_COUNT):
    """
    Export the hex map to a GeoJSON file.
    
    Args:
        centroids: List of dictionaries with hex_id, lat, lon
        filename: Output filename
    """
    # Create a GeoDataFrame with Point geometries
    gdf = gpd.GeoDataFrame(
        centroids,
        geometry=[Point(c['lon'], c['lat']) for c in centroids],
        crs="EPSG:4326"
    )
    
    # Add the hex boundaries as separate geometries
    hex_boundaries = []
    
    for hex_id in [c['hex_id'] for c in centroids]:
        # Get the boundary coordinates - using H3 v4.x API
        # In the latest H3 version, we need to manually swap the coordinates
        # to get them in the format expected by Shapely (lon, lat)
        boundary = h3.cell_to_boundary(hex_id)
        # Convert coordinates from (lat, lng) to (lng, lat) for Shapely
        boundary = [(lng, lat) for lat, lng in boundary]
        hex_boundaries.append(Polygon(boundary))
    
    # Create a GeoDataFrame with the hex polygons
    hex_gdf = gpd.GeoDataFrame(
        [{'hex_id': c['hex_id']} for c in centroids],
        geometry=hex_boundaries,
        crs="EPSG:4326"
    )
    
    # Save both to GeoJSON
    hex_gdf.to_file(f'./Data/hex_hash_map/{NETWORK_NAME}_{HUB_COUNT}/{NETWORK_NAME}_{HUB_COUNT}_hex_map.geojson', driver="GeoJSON")
    print(f"Exported hex map")
    
    # Save centroids to a separate GeoJSON
    gdf.to_file(f'./Data/hex_hash_map/{NETWORK_NAME}_{HUB_COUNT}/{NETWORK_NAME}_{HUB_COUNT}_hex_map_centroids.geojson', driver="GeoJSON")
    print(f"Exported centroid points")

def generate_bounding_box(stops_file,buffer_percentage=2):
    """
    Generate a bounding box from GTFS stops.txt file.
    
    Args:
        stops_file_path: Path to the GTFS stops.txt file
        buffer_percentage: Percentage to expand the bounding box (default: 2%)
        
    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon)
    """
    
    # Ensure the required columns exist
    if 'stop_lat' not in stops_file.columns or 'stop_lon' not in stops_file.columns:
        raise ValueError("stops.txt file must contain 'stop_lat' and 'stop_lon' columns")
    
    # Get the min and max coordinates
    min_lat = stops_file['stop_lat'].min()
    min_lon = stops_file['stop_lon'].min()
    max_lat = stops_file['stop_lat'].max()
    max_lon = stops_file['stop_lon'].max()

    # Calculate the size of the bounding box
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    # Add buffer around the bounding box
    buffer_lat = lat_range * (buffer_percentage / 100)
    buffer_lon = lon_range * (buffer_percentage / 100)
    
    # Return the bounding box with buffer
    return (
        min_lat - buffer_lat,
        min_lon - buffer_lon,
        max_lat + buffer_lat,
        max_lon + buffer_lon
    )

def main():
    import pandas as pd

    with open(f'./parameters_entered.txt', 'rb') as file:
        parameter_files = pickle.load(file)
    BUILD_TRANSFER, NETWORK_NAME, BUILD_TBTR_FILES, BUILD_TRANSFER_PATTERNS_FILES, BUILD_CSA = parameter_files
    
    HUB_COUNT = int(input(f"Enter the number of hub stops for {NETWORK_NAME}. Example: 0\n: "))
    
    resolution = int(input(f"Enter resolution between 0 to 15. Example: 9\n: "))

    if not os.path.exists(f'./logs/.'):
            os.makedirs(f'./logs/.')
    if not os.path.exists(f'./Data/hex_hash_map/{NETWORK_NAME}_{HUB_COUNT}/.'):
        os.makedirs(f'./Data/hex_hash_map/{NETWORK_NAME}_{HUB_COUNT}/.')

    stops_file = pd.read_csv(f"Data/GTFS/{NETWORK_NAME}/stops.txt")

    bounding_box = generate_bounding_box(stops_file)

    centroids = generate_hex_map(resolution, bounding_box)

    print(f"Generated {len(centroids)} hexagons at resolution {resolution}")
    print('Creating csv and json files.')
    export_to_csv(centroids, NETWORK_NAME, HUB_COUNT)
    export_to_geojson(centroids, NETWORK_NAME, HUB_COUNT)
    print('Finished.')

if __name__== "__main__":
    main()