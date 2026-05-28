import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import BallTree
import osmnx as ox
import networkx as nx
import os
import sys
import geopandas as gpd
from pyproj import Transformer
import geopandas as gpd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === Helper Functions ===
def calculate_walking_time(distance_meters, speed_kmph=4.0):
    """Convert distance in meters to walking time in seconds."""
    speed_mps = speed_kmph * 1000 / 3600  # km/h → m/s
    return distance_meters / speed_mps  # Time in seconds

def time_to_seconds(time_str):
    """Convert HH:MM:SS string to total seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))

def prepare_road_network(place_name="Rajkot, India", network_type='walk'):
    """
    Extract and prepare the road network for pedestrian routing.
    This function should be called once at the beginning.
    
    Args:
        place_name: Name of the area to extract the road network for
        network_type: Type of network to extract (walk, bike, drive, etc.)
        
    Returns:
        Dictionary containing the network graph, CRS, and transformer
    """
    print(f"Extracting road network for {place_name}...")
    
    # Extract and project the walking network
    G = ox.graph_from_place(place_name, network_type=network_type)
    G = ox.project_graph(G)
    graph_crs = G.graph['crs']
    
    # Create transformer for coordinate projection
    transformer = Transformer.from_crs(4326, graph_crs, always_xy=True)
    
    return {
        'graph': G,
        'crs': graph_crs,
        'transformer': transformer
    }

# Function to filter all stops within a circular area around a source stop
def stops_in_circle(stops, source_stop_id, bus_speed_kmph, travel_time_minutes):
    """
    Computes a circle of radius (bus_speed_kmph * (travel_time_minutes/60)) kilometers 
    from the source stop, and returns all stops within that circle.
    """
    # Compute the maximum travel distance (radius) in km and convert to meters
    radius_km = bus_speed_kmph * (travel_time_minutes / 60.0)
    radius_meters = radius_km * 1000
    earth_radius = 6371000  # in meters

    # Convert stops' coordinates to radians
    stops_coords = np.deg2rad(stops[['stop_lat', 'stop_lon']].values)
    tree = BallTree(stops_coords, metric='haversine')
    
    # Get source stop's coordinates
    source_stop = stops[stops['stop_id'] == source_stop_id]

    if source_stop.empty:
        raise ValueError("Source stop id not found in the stops data.")
    source_coord = np.deg2rad(source_stop[['stop_lat', 'stop_lon']].iloc[0].values.reshape(1, -1))
    
    # Convert the radius from meters to radians
    radius_radians = radius_meters / earth_radius
    
    # Query the tree for stops within the radius
    indices = tree.query_radius(source_coord, r=radius_radians)[0]
    stops_in_circle = stops.iloc[indices].copy()

    return stops_in_circle


def stops_in_circle_walk(stops, user_location, bus_speed_kmph, travel_time_minutes):
    """
    Computes a circle of radius (bus_speed_kmph * (travel_time_minutes/60)) kilometers 
    from the user location, and returns all stops within that circle.
    
    Args:
        stops: DataFrame with stop_id, stop_name, stop_lat, stop_lon
        user_location: Tuple of (lat, lon) representing user's location
        bus_speed_kmph: Bus speed in km/h
        travel_time_minutes: Maximum travel time in minutes
        
    Returns:
        DataFrame containing stops within the circle
    """
    # Compute the maximum travel distance (radius) in km and convert to meters
    radius_km = bus_speed_kmph * (travel_time_minutes / 60.0)
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


def find_stops_within_time_threshold(
    stops_df, 
    stoptimes_df,
    user_location, 
    start_time,
    walking_time_threshold_minutes=10,
    network_data=None,
    output_file=None,
    walking_speed_kmph=4.0  
):
    """
    Find all stops reachable within a specified walking time threshold using network distances.
    Uses a pre-loaded network graph for efficiency.
    
    Args:
        stops_df: DataFrame with stop_id, stop_name, stop_lat, stop_lon
        stoptimes_df: DataFrame with stop_id, departure_time
        user_location: Tuple of (lat, lon)
        start_time: User's start time as a datetime.time object
        walking_time_threshold_minutes: Maximum walking time in minutes
        network_data: Dictionary containing graph, CRS, and transformer
        output_file: Path to output file
        walking_speed_kmph: Walking speed in km/h
        buffer_speed_kmph: Speed used for initial buffer calculation
        buffer_multiplier: Multiplier to expand buffer beyond exact threshold
    """
    # Convert walking time threshold to seconds
    walking_time_threshold_seconds = walking_time_threshold_minutes * 60
    
    # Load transfers data and ensure correct time units
    transfers_df = pd.read_csv('Data/GTFS/rajkot/transfers.csv')
    if transfers_df['min_transfer_time'].max() < 100:  # Assume minutes if < 100
        transfers_df['min_transfer_time'] *= 60  # Convert minutes → seconds

    # Convert start time to seconds
    start_time_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second

    # STEP 1: Find all stops within buffer area (using faster-than-walking speed)
    # This creates a larger area to ensure we don't miss any potential stops
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
    
    # Find nearest stop via network distance
    print(f"Calculating actual walking times for {len(buffer_stops_df_with_proj)} stops in buffer...")
    nearest_stop_id = None
    nearest_stop_distance = float('inf')
    
    # Store all direct distances
    direct_distances = {}
    
    for _, stop in buffer_stops_df_with_proj.iterrows():
        stop_node = ox.distance.nearest_nodes(G, stop.x_proj, stop.y_proj)
        try:
            distance = nx.shortest_path_length(G, user_node, stop_node, weight='length')
            direct_distances[stop.stop_id] = distance
            if distance < nearest_stop_distance:
                nearest_stop_distance = distance
                nearest_stop_id = stop.stop_id
        except nx.NetworkXNoPath:
            continue

    if nearest_stop_id is None:
        print("No reachable stops found.")
        return []
    
    print(f"Nearest stop: {nearest_stop_id}, Distance: {nearest_stop_distance:.2f}m")

    # Use the pre-built transfer graph if available, otherwise build it
    if 'transfer_graph' in network_data:
        transfer_graph = network_data['transfer_graph']
    else:
        # Build transfer graph
        transfer_graph = nx.Graph()
        for _, row in transfers_df.iterrows():
            transfer_graph.add_edge(
                row['from_stop_id'],
                row['to_stop_id'],
                weight=row['min_transfer_time']
            )
        
        # Store for future use
        network_data['transfer_graph'] = transfer_graph
    
    # Calculate walking times for all stops (direct walking or walking + transfer)
    walking_times = {}
    
    # First, add direct walking times for ALL stops (if possible)
    for stop_id, distance in direct_distances.items():
        walking_time = calculate_walking_time(distance, walking_speed_kmph)
        # Only include if within the walking time threshold
        if walking_time <= walking_time_threshold_seconds:
            walking_times[stop_id] = walking_time
    
    # Check if transfers provide better times than direct walking
    if nearest_stop_id in transfer_graph.nodes:
        # Calculate walking time to nearest stop
        user_walking_time = calculate_walking_time(nearest_stop_distance, walking_speed_kmph)
        
        try:
            paths = nx.single_source_dijkstra_path_length(
                transfer_graph, nearest_stop_id, weight='weight'
            )
            
            # For each transfer option, see if it's faster than direct walking
            for stop_id, transfer_time in paths.items():
                if stop_id in buffer_stops_df_with_proj['stop_id'].values:  # Only consider stops in buffer
                    total_time = user_walking_time + transfer_time
                    
                    # Only update if transfer is faster than direct walking AND within threshold
                    if total_time <= walking_time_threshold_seconds and (
                        stop_id not in walking_times or total_time < walking_times[stop_id]
                    ):
                        walking_times[stop_id] = total_time
        
        except nx.NetworkXError as e:
            print(f"Transfer graph error: {e}")
    else:
        print(f"Stop {nearest_stop_id} not in transfer graph.")

    # Process stop times
    stoptimes_df['departure_time'] = pd.to_datetime(stoptimes_df['departure_time'])
    
    # Get actual date from stoptimes
    base_date = stoptimes_df['departure_time'].dt.date.min()
    base_datetime = pd.Timestamp(base_date)

    # Calculate arrival/waiting times
    stop_results = []
    
    for stop_id, walk_time in walking_times.items():
        # User arrival time at this stop
        arrival_time = base_datetime + pd.Timedelta(seconds=start_time_seconds + walk_time)
        
        # Filter departures for this stop that occur after user arrival
        valid_departures = stoptimes_df[
            (stoptimes_df['stop_id'] == stop_id) &
            (stoptimes_df['departure_time'] >= arrival_time)
        ]
        
        if valid_departures.empty:
            continue
        
        # Get earliest departure after arrival
        next_departure = valid_departures['departure_time'].min()
        
        # Convert to seconds from day start
        departure_sec = (next_departure.hour * 3600 + 
                        next_departure.minute * 60 + 
                        next_departure.second)
        
        # Calculate waiting time
        waiting_time = departure_sec - (start_time_seconds + walk_time)
        
        if waiting_time >= 0:
            stop_results.append({
                'stop_id': stop_id,
                'walking_time': walk_time,
                'waiting_time': waiting_time,
                'total_time': walk_time + waiting_time
            })
    
    # Sort by total time (walking + waiting)
    stop_results.sort(key=lambda x: x['total_time'])

    print(f"Found {len(stop_results)} reachable stops with valid departures.")
    
    # Write results to file
    if output_file:
        with open(output_file, 'w') as f:
            for result in stop_results:
                stop_id = result['stop_id']
                stop_info = stops_df[stops_df.stop_id == stop_id]
                
                if not stop_info.empty:
                    stop_name = stop_info.iloc[0]['stop_name']
                    f.write(
                        f"Stop_ID: {stop_id}, "
                        f"Stop_Name: {stop_name}, "
                        f"Walking: {seconds_to_time(int(result['walking_time']))}, "
                        f"Waiting: {seconds_to_time(int(result['waiting_time']))}, "
                        f"Total: {seconds_to_time(int(result['total_time']))}\n"
                    )
    return stop_results