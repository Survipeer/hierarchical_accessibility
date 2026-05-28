import pandas as pd
import osmnx as ox
import networkx as nx
from pyproj import Transformer
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


# ================= GLOBALS (same pattern as your script) =================
GLOBAL_HEX_DF = None
GLOBAL_NETWORK_DATA = None
GLOBAL_HEX_TREE = None
GLOBAL_HEX_COORDS = None


# ================= HELPER FUNCTIONS =================

def calculate_walking_time_seconds(distance_meters, speed_kmph=4.0):
    """Return walking time in seconds"""
    speed_mps = speed_kmph * 1000 / 3600
    return distance_meters / speed_mps


def prepare_road_network(place_name, network_type='walk'):
    graph_dir = "Data/graphs"
    os.makedirs(graph_dir, exist_ok=True)

    filename = f"{place_name.replace(', ', '_').replace(' ', '_')}_walk_graph.gpickle"
    graph_path = os.path.join(graph_dir, filename)

    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
    else:
        G = ox.graph_from_place(place_name, network_type=network_type)
        G = ox.project_graph(G)
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)

    crs = G.graph['crs']
    transformer = Transformer.from_crs(4326, crs, always_xy=True)

    return {
        "graph": G,
        "crs": crs,
        "transformer": transformer
    }


def hexes_in_circle(origin_lat, origin_lon, walking_speed_kmph, walking_time_minutes):
    """Fast circle prune using global BallTree"""
    radius_km = walking_speed_kmph * (walking_time_minutes / 60.0)
    radius_m = radius_km * 1000
    earth_radius = 6371000

    origin_rad = np.deg2rad([[origin_lat, origin_lon]])
    radius_rad = radius_m / earth_radius

    indices = GLOBAL_HEX_TREE.query_radius(origin_rad, r=radius_rad)[0]
    return GLOBAL_HEX_DF.iloc[indices]


def find_hexes_within_walking_distance(
    hex_df,
    origin_hex_id,
    walking_time_threshold_minutes,
    network_data,
    walking_speed_kmph
):
    origin = hex_df.loc[origin_hex_id]
    origin_lat, origin_lon = origin.lat, origin.lon

    buffer_hexes = hexes_in_circle(
        origin_lat,
        origin_lon,
        walking_speed_kmph,
        walking_time_threshold_minutes
    )

    if buffer_hexes.empty:
        return []

    G = network_data["graph"]
    transformer = network_data["transformer"]

    # snap origin to OSM node
    x0, y0 = transformer.transform(origin_lon, origin_lat)
    origin_node = ox.distance.nearest_nodes(G, x0, y0)

    # max walkable distance in meters
    max_dist_m = walking_speed_kmph * 1000 * (walking_time_threshold_minutes / 60)

    # single-source Dijkstra
    lengths = nx.single_source_dijkstra_path_length(
        G,
        origin_node,
        cutoff=max_dist_m,
        weight="length"
    )

    results = []

    for hex_id, row in buffer_hexes.iterrows():
        if hex_id == origin_hex_id:
            continue

        x, y = transformer.transform(row.lon, row.lat)
        dest_node = ox.distance.nearest_nodes(G, x, y)

        if dest_node not in lengths:
            continue

        dist = lengths[dest_node]
        walking_time_sec = calculate_walking_time_seconds(dist, walking_speed_kmph)

        if walking_time_sec <= walking_time_threshold_minutes * 60:
            results.append({
                "hex_id": hex_id,
                "walking_time_minutes": round(walking_time_sec / 60, 2),
                "walking_distance_meters": round(dist, 1)
            })

    results.sort(key=lambda x: x["walking_time_minutes"])
    return results


def process_hex_wrapper(args):
    hex_id, walking_time_threshold_minutes, walking_speed_kmph = args

    try:
        walkable_hexes = find_hexes_within_walking_distance(
            GLOBAL_HEX_DF,
            hex_id,
            walking_time_threshold_minutes,
            GLOBAL_NETWORK_DATA,
            walking_speed_kmph
        )

        row = GLOBAL_HEX_DF.loc[hex_id]

        return {
            hex_id: {
                "location": {
                    "lat": row.lat,
                    "lon": row.lon
                },
                "walking_threshold_minutes": walking_time_threshold_minutes,
                "walking_speed_kmph": walking_speed_kmph,
                "walkable_hexes": walkable_hexes,
                "total_hexes_found": len(walkable_hexes)
            }
        }

    except Exception as e:
        return {
            hex_id: {
                "error": str(e)
            }
        }


def init_worker(hex_df_, network_data_):
    """Initialize globals per worker (same style as your original script)"""
    global GLOBAL_HEX_DF, GLOBAL_NETWORK_DATA
    global GLOBAL_HEX_TREE, GLOBAL_HEX_COORDS

    GLOBAL_HEX_DF = hex_df_
    GLOBAL_NETWORK_DATA = network_data_

    GLOBAL_HEX_COORDS = np.deg2rad(hex_df_[['lat', 'lon']].values)
    GLOBAL_HEX_TREE = BallTree(GLOBAL_HEX_COORDS, metric='haversine')


# ================= MAIN =================

if __name__ == "__main__":
    with open('./parameters_entered.txt', 'rb') as file:
        parameter_files = pickle.load(file)

    BUILD_TRANSFER, NETWORK_NAME, BUILD_TBTR_FILES, BUILD_TRANSFER_PATTERNS_FILES, BUILD_CSA = parameter_files
    HUB_COUNT = int(input(f"Enter the number of hub stops for {NETWORK_NAME}. Example: 0\n: "))

    csv_file_path = f"./Data/hex_hash_map/{NETWORK_NAME}_{HUB_COUNT}/{NETWORK_NAME}_{HUB_COUNT}_hex_centroid.csv"

    output_dir = f"./Data/walk_only_hex2hex/{NETWORK_NAME}_{HUB_COUNT}"
    os.makedirs(output_dir, exist_ok=True)

    output_json = f"{output_dir}/{NETWORK_NAME}_{HUB_COUNT}_walk_only_hex2hex.json"

    place_name = f"{NETWORK_NAME}, India"
    walking_time_threshold_minutes = int(input("Enter walking threshold time in minutes: "))
    walking_speed_kmph = 4.0

    USE_PARALLEL = int(input("Hex2Hex walk can be built in parallel. Enter 1 to use multiprocessing. Else press 0\n: "))
    CORES = 0
    if USE_PARALLEL == 1:
        CORES = int(input(f"Enter number of CORES (>=1). Available: {multiprocessing.cpu_count()}\n: "))

    start = time()

    print("Loading OSM walking graph...")
    network_data = prepare_road_network(place_name)

    print("Loading hex centroids...")
    hex_df = pd.read_csv(csv_file_path).set_index("hex_id")

    args_list = [
        (hex_id, walking_time_threshold_minutes, walking_speed_kmph)
        for hex_id in hex_df.index
    ]

    results = {}

    if USE_PARALLEL == 1:
        ctx = get_context("spawn")
        with ctx.Pool(
            CORES,
            initializer=init_worker,
            initargs=(hex_df, network_data)
        ) as pool:
            for partial in tqdm(
                pool.imap_unordered(process_hex_wrapper, args_list),
                total=len(args_list),
                desc="Processing hexes (parallel)"
            ):
                results.update(partial)
    else:
        init_worker(hex_df, network_data)
        for args in tqdm(args_list, desc="Processing hexes (sequential)"):
            results.update(process_hex_wrapper(args))

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Saved to {output_json}")
    print(f"Time taken: {round((time() - start)/60, 2)} mins", file=sys.__stdout__)
