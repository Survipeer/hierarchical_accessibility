"""
reachability_hierarchical.py

Runs the hierarchical hex-to-hex accessibility analysis.
One output JSON per hex centroid, format identical to the original
reachability_analysis_hex2hex.py so downstream tools work unchanged.

Changes from the old version
------------------------------
- data['time']      -> data['total_time']       (key rename from new stitcher)
- data['inter_walk']-> removed (stitcher no longer returns this;
                       stop-to-stop walking is internal to the chunk model)
- walking breakdown now uses ingress_walk + egress_walk only
  (ingress_transit and egress_transit are transit legs, not walking)
- CORES and TIME_INTERVALS are now user inputs instead of hardcoded
- max_walking and max_total_time are passed through to the stitcher
"""

import json
import os
import sys
import pickle
import datetime
import multiprocessing
from multiprocessing import get_context
from collections import defaultdict
from time import time

from tqdm import tqdm

from hierarchical_query import HierarchicalStitcher

# ---------------------------------------------------------------------------
# Worker globals
# ---------------------------------------------------------------------------
STITCHER      = None
WALKABLE_STOPS = None
STOP_TO_HEX   = None
TIME_INTERVALS = None
NETWORK_NAME  = None
WALK_ONLY_OSM = None
MAX_WALKING    = None
MAX_TOTAL_TIME = None
MAX_TRANSFER   = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_bucket(t):
    for a, b in ((0, 15), (15, 30), (30, 45), (45, 60)):
        if a < t <= b:
            return f"{a}-{b} min"
    return None


# ---------------------------------------------------------------------------
# Per-hex worker
# ---------------------------------------------------------------------------

def process_hex_centroid_hierarchical(hex_task):
    origin_hex_id, hex_data = hex_task

    # 1. ---- WALK-ONLY BASELINE ----
    best_by_walk = {}
    for w in WALK_ONLY_OSM.get(origin_hex_id, {}).get("walkable_hexes", []):
        dest_hex   = w["hex_id"]
        total_walk = w["walking_time_minutes"]
        if total_walk <= MAX_WALKING:
            best_by_walk[dest_hex] = {
                "hex_id":             dest_hex,
                "total_walking_time": round(total_walk, 2),
            }

    # 2. ---- HIERARCHICAL TRANSIT REACHABILITY ----
    results_by_time = {}

    for t_interval in TIME_INTERVALS:
        t_str = t_interval.strftime("%Y-%m-%d %H:%M:%S")

        # get_stitched_results returns:
        # {dest_stop_id(int): {
        #     'total_time': float,       # minutes from departure to dest stop
        #     'transfers': int,
        #     'ingress_walk': float,     # hex -> boarding stop (minutes)
        #     'ingress_transit': float,  # boarding stop -> src gate (minutes)
        #     'egress_transit': float,   # dest gate -> dest stop (minutes)
        # }}
        stop_results = STITCHER.get_stitched_results(
            origin_stops   = hex_data["walkable_stops"],
            departure_time = t_interval,
            max_walking    = MAX_WALKING,
            max_total_time = MAX_TOTAL_TIME,
            max_transfer   = MAX_TRANSFER,
        )

        # Use dicts keyed by dest_hex to keep only the best entry per hex
        # per bucket — exactly like the original best_by_transit logic.
        # Multiple destination stops can map to the same hex; we keep the
        # one with the lowest total_time.
        bucket_best = {
            "0-15 min":  {},
            "15-30 min": {},
            "30-45 min": {},
            "45-60 min": {},
        }

        for stop_id, data in stop_results.items():
            # Map destination stop -> destination hex(es)
            for dest_hex, egress_walk_min in STOP_TO_HEX.get(int(stop_id), []):
                if dest_hex == origin_hex_id:
                    continue

                # total_time already covers hex->stop->gate->gate->stop
                # add the final stop->hex walk on the destination side
                total_time = data["total_time"] + egress_walk_min

                bucket = assign_bucket(total_time)
                if not bucket:
                    continue

                # Walking breakdown:
                #   hex_to_stop = ingress walk (origin side)
                #   stop_to_hex = egress walk  (destination side)
                #   ingress and egress transit legs are not walking
                hex_to_stop   = data.get("ingress_walk", 0)
                stop_to_hex   = egress_walk_min
                total_walking = hex_to_stop + stop_to_hex

                # Enforce global walking budget across both walk legs
                if total_walking > MAX_WALKING:
                    continue

                # Enforce global total time (egress walk may push it over)
                if total_time > MAX_TOTAL_TIME:
                    continue

                entry = {
                    "hex_id":                    dest_hex,
                    "destination_stop_id":       int(stop_id),
                    "transfers":                 data["transfers"],
                    "total_time":                round(total_time, 2),
                    "hex_to_stop_walking_time":  round(hex_to_stop, 2),
                    "stop_to_stop_walking_time": 0.0,  # absorbed into transit chunks
                    "stop_to_hex_walking_time":  round(stop_to_hex, 2),
                    "total_walking_time":        round(total_walking, 2),
                }

                # Keep only the best (lowest total_time) entry per dest_hex
                existing = bucket_best[bucket].get(dest_hex)
                if existing is None or total_time < existing["total_time"]:
                    bucket_best[bucket][dest_hex] = entry

        # Convert dicts to lists for JSON output
        buckets = {
            b: {"destination": list(best.values())}
            for b, best in bucket_best.items()
        }

        results_by_time[t_str] = buckets

    # 3. ---- SAVE ----
    out_file = (f"./hex_to_hex_results/{NETWORK_NAME}_hierarchical/"
                f"{origin_hex_id}_results.json")
    with open(out_file, "w") as f:
        json.dump(
            {
                "origin_hex": origin_hex_id,
                "walk_results": {
                    "destination": list(best_by_walk.values())
                },
                "results_by_time": results_by_time,
            },
            f, indent=2,
        )


# ---------------------------------------------------------------------------
# Worker initialiser (called once per spawned process)
# ---------------------------------------------------------------------------

def init_worker(net_name, intervals, stop_map, walkable_data,
                walk_only_osm, max_walking, max_total_time, max_transfer):
    global STITCHER, TIME_INTERVALS, STOP_TO_HEX, WALKABLE_STOPS
    global NETWORK_NAME, WALK_ONLY_OSM, MAX_WALKING, MAX_TOTAL_TIME, MAX_TRANSFER

    NETWORK_NAME   = net_name
    STITCHER       = HierarchicalStitcher(net_name)
    TIME_INTERVALS = intervals
    STOP_TO_HEX    = stop_map
    WALKABLE_STOPS = walkable_data
    WALK_ONLY_OSM  = walk_only_osm
    MAX_WALKING    = max_walking
    MAX_TOTAL_TIME = max_total_time
    MAX_TRANSFER   = max_transfer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    with open("./parameters_entered.txt", "rb") as f:
        parameter_files = pickle.load(f)
    BUILD_TRANSFER, NETWORK_NAME, BUILD_TBTR_FILES, \
        BUILD_TRANSFER_PATTERNS_FILES, BUILD_CSA = parameter_files

    # --- Inputs ---
    HUB_COUNT = int(input(
        f"Enter hub stop count for {NETWORK_NAME} (Example: 0)\n: "))

    default_date = "2025-01-01"
    base_date = input(
        f"Enter analysis date (YYYY-MM-DD, default: {default_date})\n: ").strip()
    if not base_date:
        base_date = default_date

    MAX_WALKING    = float(input("Max walking time in minutes (default: 10)\n: ") or 10)
    MAX_TOTAL_TIME = float(input("Max total journey time in minutes (default: 60)\n: ") or 60)
    MAX_TRANSFER   = int(input("Max transfers globally across all phases (default: 3)\n: ") or 3)

    USE_PARALLEL = int(input(
        "Run in parallel? 1=yes, 0=no (default: 0)\n: ") or 0)
    CORES = 1
    if USE_PARALLEL:
        CORES = int(input(
            f"Number of cores (available: {multiprocessing.cpu_count()})\n: "))

    # --- Load walkable stops ---
    ws_path = (f"./Data/walkable_stops/{NETWORK_NAME}_{HUB_COUNT}/"
               f"{NETWORK_NAME}_{HUB_COUNT}_walkable_stops.json")
    with open(ws_path) as f:
        walkable_data = json.load(f)

    # --- Load OSM walk-only baseline ---
    osm_path = (f"./Data/walk_only_hex2hex/{NETWORK_NAME}_{HUB_COUNT}/"
                f"{NETWORK_NAME}_{HUB_COUNT}_walk_only_hex2hex.json")
    print("Loading OSM walk-only baseline...")
    with open(osm_path) as f:
        WALK_ONLY_OSM = dict(json.load(f))

    # --- Build stop -> hex mapping ---
    stop_to_hex = defaultdict(list)
    for h_id, h_data in walkable_data.items():
        for s in h_data.get("walkable_stops", []):
            stop_to_hex[s["stop_id"]].append(
                (h_id, s["walking_time_minutes"]))

    # --- Time intervals ---
    start_dt = datetime.datetime.strptime(f"{base_date} 00:00", "%Y-%m-%d %H:%M")
    end_dt   = datetime.datetime.strptime(f"{base_date} 23:45", "%Y-%m-%d %H:%M")
    TIME_INTERVALS = []
    cur = start_dt
    while cur <= end_dt:
        TIME_INTERVALS.append(cur)
        cur += datetime.timedelta(minutes=15)

    print(f"Time intervals: {len(TIME_INTERVALS)} "
          f"({start_dt.strftime('%H:%M')} — {end_dt.strftime('%H:%M')}, 15 min steps)")

    # --- Output directory ---
    out_dir = f"./hex_to_hex_results/{NETWORK_NAME}_hierarchical"
    os.makedirs(out_dir, exist_ok=True)

    hex_tasks = list(walkable_data.items())
    print(f"\nStarting hierarchical analysis: "
          f"{len(hex_tasks)} hexes, {CORES} core(s)...")

    t1 = time()

    if USE_PARALLEL and CORES > 1:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes  = CORES,
            initializer= init_worker,
            initargs   = (NETWORK_NAME, TIME_INTERVALS, stop_to_hex,
                          walkable_data, WALK_ONLY_OSM,
                          MAX_WALKING, MAX_TOTAL_TIME, MAX_TRANSFER),
        ) as pool:
            list(tqdm(
                pool.imap_unordered(process_hex_centroid_hierarchical, hex_tasks),
                total=len(hex_tasks),
                desc="Hexes",
            ))
    else:
        # Sequential — useful for debugging
        init_worker(NETWORK_NAME, TIME_INTERVALS, stop_to_hex,
                    walkable_data, WALK_ONLY_OSM,
                    MAX_WALKING, MAX_TOTAL_TIME, MAX_TRANSFER)
        for task in tqdm(hex_tasks, desc="Hexes"):
            process_hex_centroid_hierarchical(task)

    print(f"\nCompleted in {round((time() - t1) / 60, 2)} minutes.")
    print(f"Results in: {out_dir}/")