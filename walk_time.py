import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from pyproj import Transformer
from datetime import datetime
import numpy as np
import json
import os
from sklearn.neighbors import BallTree
import pickle
from tqdm import tqdm
import sys
import multiprocessing
from multiprocessing import get_context
from time import time

GLOBAL_STOPS_DF = None
GLOBAL_NETWORK_DATA = None
GLOBAL_OUTPUT_JSON = None


# === Helper Functions ===
def calculate_walking_time(distance_meters, speed_kmph=4.0):
    """Convert distance in meters to walking time in seconds."""
    speed_mps = speed_kmph * 1000 / 3600  # km/h → m/s
    return distance_meters / speed_mps  # Time in seconds

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def prepare_road_network(place_name, network_type='walk'):
    """
    Load road network from disk if it exists, else download, project, and save it.
    """
    graph_dir = "Data/graphs"
    os.makedirs(graph_dir, exist_ok=True)

    # Normalize filename
    filename = f"{place_name.replace(', ', '_').replace(' ', '_')}_walk_graph.gpickle"
    graph_path = os.path.join(graph_dir, filename)

    if os.path.exists(graph_path):
        print(f"Loading road graph from {graph_path}")
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

    else:
        print(f"Downloading and building road graph for {place_name}...")
        G = ox.graph_from_place(place_name, network_type=network_type)
        G = ox.project_graph(G)
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)

        print(f"Graph saved to {graph_path}")

    crs = G.graph['crs']
    transformer = Transformer.from_crs(4326, crs, always_xy=True)

    return {
        'graph': G,
        'crs': crs,
        'transformer': transformer
    }


def stops_in_circle_walk(stops, user_location, walking_speed_kmph, walking_time_minutes):
    """Compute a circle around user location and return stops within that circle."""
    # Compute the maximum travel distance (radius) in km and convert to meters
    radius_km = walking_speed_kmph * (walking_time_minutes / 60.0)
    radius_meters = radius_km * 1000
    earth_radius = 6371000  # in meters

    # Convert stops' coordinates to radians
    stops_coords = np.deg2rad(stops[['stop_lat', 'stop_lon']].values)
    tree = BallTree(stops_coords, metric='haversine')
    
    # Get user's coordinates and convert to radians
    user_lat, user_lon = user_location
    user_coord = np.deg2rad([[user_lat, user_lon]])
    
    # Convert the radius from meters to radians
    radius_radians = radius_meters / earth_radius
    
    # Query the tree for stops within the radius
    indices = tree.query_radius(user_coord, r=radius_radians)[0]
    stops_in_buffer = stops.iloc[indices].copy()
    
    print(f"Found {len(stops_in_buffer)} stops within {radius_km:.2f} km buffer.")
    return stops_in_buffer

def find_stops_within_walking_distance(
    stops_df, 
    user_location, 
    walking_time_threshold_minutes=5,
    network_data=None,
    walking_speed_kmph=4.0
):
    """Find all stops reachable within walking time threshold with network distances."""
    # Convert walking time threshold to seconds
    walking_time_threshold_seconds = walking_time_threshold_minutes * 60
    
    # STEP 1: Find all stops within buffer area
    buffer_time_minutes = walking_time_threshold_minutes
    stops_in_buffer = stops_in_circle_walk(
        stops_df, 
        user_location, 
        walking_speed_kmph, 
        buffer_time_minutes
    )
    
    # If no stops found in buffer, return empty list
    if stops_in_buffer.empty:
        print("No stops found within buffer area.")
        return []
    
    # STEP 2: Calculate actual walking distances only for stops in buffer
    # Extract network elements from the pre-loaded data
    G = network_data['graph']
    graph_crs = network_data['crs']
    transformer = network_data['transformer']
    
    # Project user coordinates using the pre-loaded transformer
    user_lon, user_lat = user_location[1], user_location[0]
    user_x, user_y = transformer.transform(user_lon, user_lat)

    # Find nearest node for user
    user_node = ox.distance.nearest_nodes(G, user_x, user_y)

    # Project all bus stops in buffer to graph's CRS
    buffer_stops_gdf = gpd.GeoDataFrame(
        stops_in_buffer,
        geometry=gpd.points_from_xy(stops_in_buffer.stop_lon, stops_in_buffer.stop_lat, crs=4326)
    ).to_crs(graph_crs)
    
    # Add projected coordinates to the stops DataFrame
    buffer_stops_df_with_proj = stops_in_buffer.copy()
    buffer_stops_df_with_proj['x_proj'] = buffer_stops_gdf.geometry.x
    buffer_stops_df_with_proj['y_proj'] = buffer_stops_gdf.geometry.y
    
    # Find walking distances to all stops
    print(f"Calculating actual walking times for {len(buffer_stops_df_with_proj)} stops in buffer...")
    
    # Store walking times for each stop
    stop_results = []
    
    for _, stop in buffer_stops_df_with_proj.iterrows():
        stop_node = ox.distance.nearest_nodes(G, stop.x_proj, stop.y_proj)
        try:
            # Calculate network distance (along roads)
            distance = nx.shortest_path_length(G, user_node, stop_node, weight='length')
            
            # Convert distance to walking time
            walking_time = calculate_walking_time(distance, walking_speed_kmph)
            
            # Only include if within the walking time threshold
            if walking_time <= walking_time_threshold_seconds:
                stop_results.append({
                    'stop_id': stop.stop_id,
                    'stop_name': stop.stop_name,
                    'stop_lat': stop.stop_lat,
                    'stop_lon': stop.stop_lon,
                    'walking_time': walking_time,
                    'walking_distance': distance
                })
        except nx.NetworkXNoPath:
            # Skip stops that can't be reached via the network
            continue
    
    # Sort by walking time
    stop_results.sort(key=lambda x: x['walking_time'])

    print(f"Found {len(stop_results)} reachable stops within walking threshold.")
    return stop_results

def process_csv_locations(hex_id, lat, lon,stops_df,walking_time_threshold_minutes,network_data,walking_speed_kmph):
    """
    Process all locations in a CSV file and save walking times to a JSON file.
    
    Args:
        csv_file_path: Path to CSV with hex_id, lat, lon columns
        gtfs_folder: Path to GTFS data folder
        output_json: Path to output JSON file
        place_name: Name of the area for road network extraction
        walking_time_threshold_minutes: Maximum walking time in minutes
        walking_speed_kmph: Walking speed in km/h
    """
    
    user_location = (lat, lon)
        
    print(f"Processing location {hex_id} at ({lat}, {lon})...")
        
    # Process this location
    try:
        walkable_stops = find_stops_within_walking_distance(
            stops_df=stops_df,
            user_location=user_location,
            walking_time_threshold_minutes=walking_time_threshold_minutes,
            network_data=network_data,
            walking_speed_kmph=walking_speed_kmph
        )
            
        # Convert results to JSON-serializable format
        serializable_stops = []
        for stop in walkable_stops:
            serializable_stops.append({
                'stop_id': stop['stop_id'],
                'stop_name': stop['stop_name'],
                'stop_lat': float(stop['stop_lat']),
                'stop_lon': float(stop['stop_lon']),
                'walking_time_minutes': float(stop['walking_time'])/60.0,
                'walking_distance_meters': float(stop['walking_distance'])
            })
                      
        print(f"Found {len(serializable_stops)} walkable stops for location {hex_id}")
            
    except Exception as e:
        return {
            hex_id: {
                'location': {
                    'lat': lat,
                    'lon': lon
                },
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e)
            }
        }

    
    return {
        hex_id: {
            'location': {
                'lat': lat,
                'lon': lon
            },
            'walking_threshold_minutes': walking_time_threshold_minutes,
            'walking_speed_kmph': walking_speed_kmph,
            'walkable_stops': serializable_stops,
            'total_stops_found': len(serializable_stops)
        }
    }

def process_csv_locations_wrapper(args):
    """
    Wrapper for multiprocessing-safe processing of locations.
    """
    hex_id, lat, lon, walking_time_threshold_minutes, walking_speed_kmph = args
    return process_csv_locations(
        hex_id, lat, lon,
        GLOBAL_STOPS_DF,
        walking_time_threshold_minutes,
        GLOBAL_NETWORK_DATA,
        walking_speed_kmph
    )

def init_worker(stops_df_, network_data_, output_json_, log_path=None):
    global GLOBAL_OUTPUT_JSON, GLOBAL_STOPS_DF, GLOBAL_NETWORK_DATA

    GLOBAL_STOPS_DF = stops_df_
    GLOBAL_NETWORK_DATA = network_data_
    GLOBAL_OUTPUT_JSON = output_json_
    if log_path:
        sys.stdout = open(log_path, 'a')  # append mode




if __name__ == "__main__":
    with open(f'./parameters_entered.txt', 'rb') as file:
        parameter_files = pickle.load(file)
    BUILD_TRANSFER, NETWORK_NAME, BUILD_TBTR_FILES, BUILD_TRANSFER_PATTERNS_FILES, BUILD_CSA = parameter_files

    HUB_COUNT = int(input(f"Enter the number of hub stops for {NETWORK_NAME}. Example: 0\n: "))
    
    csv_file_path = f"./Data/hex_hash_map/{NETWORK_NAME}_{HUB_COUNT}/{NETWORK_NAME}_{HUB_COUNT}_hex_centroid.csv"
    gtfs_folder = f"Data/GTFS/{NETWORK_NAME}"
    log_path = f'./logs/walk_time_{NETWORK_NAME}_{HUB_COUNT}'
    if not os.path.exists(f'./Data/walkable_stops/{NETWORK_NAME}_{HUB_COUNT}/.'):
            os.makedirs(f'./Data/walkable_stops/{NETWORK_NAME}_{HUB_COUNT}/.')

    output_json = f"./Data/walkable_stops/{NETWORK_NAME}_{HUB_COUNT}/{NETWORK_NAME}_{HUB_COUNT}_walkable_stops.json"
    place_name = f"{NETWORK_NAME}, India"
    walking_time_threshold_minutes = int(input(f"Enter walking threshold time in minutes Example: 5\n: "))
    walking_speed_kmph = 4.0
    USE_PARALlEL = int(input("Walk can be built in parallel. Enter 1 to use multiprocessing. Else press 0. Example: 0\n: "))
    CORES = 0
    if USE_PARALlEL != 0:
        CORES = int(input(f"Enter number of CORES (>=1). \nAvailable CORES (logical and physical):  {multiprocessing.cpu_count()}\n: "))
    GENERATE_LOGFILE = int(input("Press 1 to generate logfile else press 0. Example: 0\n: "))
    if not os.path.exists(f'./logs/.'):
            os.makedirs(f'./logs/.')
    
    if GENERATE_LOGFILE == 1:
        print("All outputs will be redirected to log file")
        sys.stdout = open(f'./logs/walk_time_{NETWORK_NAME}_{HUB_COUNT}', 'w')
    
    start_time = time()
    GLOBAL_NETWORK_DATA = prepare_road_network(place_name=place_name, network_type='walk')
    df = pd.read_csv(csv_file_path)
    
    # Validate that required columns exist
    required_columns = ['hex_id', 'lat', 'lon']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    
    # Initialize or load existing results dictionary
    results = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r') as f:
                results = json.load(f)
            print(f"Loaded existing results for {len(results)} locations")
        except json.JSONDecodeError:
            print(f"Existing JSON file is corrupted. Starting with empty results.")
    
    # Process each location
    total_locations = len(df)
    print(f"Processing {total_locations} locations...")
    row_list = [row for row in df.itertuples()]
    args_list = [(row.hex_id, row.lat, row.lon, walking_time_threshold_minutes, walking_speed_kmph) for row in df.itertuples()]

    if USE_PARALlEL == 1:
        ctx = get_context("spawn")  # safer on Windows
        GLOBAL_STOPS_DF = pd.read_csv(f"{gtfs_folder}/stops.csv")
        with ctx.Pool(CORES, initializer=init_worker, initargs=(GLOBAL_STOPS_DF, GLOBAL_NETWORK_DATA, output_json, log_path)) as pool:
            all_results = list(tqdm(pool.imap_unordered(process_csv_locations_wrapper, args_list),
                        total=len(args_list),
                        desc="Processing locations (parallel)"))

            # Merge all individual dicts into one final results dict
            for partial_result in all_results:
                results.update(partial_result)

            # Save final results
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)

    else:
        GLOBAL_STOPS_DF = pd.read_csv(f"{gtfs_folder}/stops.csv")
        init_worker(GLOBAL_STOPS_DF, GLOBAL_NETWORK_DATA, output_json)

        for args in tqdm(args_list, desc="Processing locations (sequential)"):
            partial_result = process_csv_locations_wrapper(args)
            if partial_result:
                results.update(partial_result)

        # Save final results
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)


    
    print(f"\nAll locations processed. Results saved to {output_json}")
    
    print(f"Processing complete. Time taken: {round((time() - start_time)/60, 2)} mins.", file=sys.__stdout__)

    

