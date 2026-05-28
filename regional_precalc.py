"""
regional_precalc.py  —  Precompute all five chunk types.

Produces five files in ./Data/Hierarchical/{NETWORK_NAME}/:

ingress_chunks.pkl
    {stop_id: {time_bucket: {gate_id: {travel_time_minutes, transfers}}}}
    Every stop (including gates) in a region to every gate in the SAME region.
    Used in Phase 1 of query-time stitching to move a passenger from their
    boarding stop to the nearest region boundary gate.

egress_chunks.pkl
    {gate_id: {time_bucket: {stop_id: {travel_time_minutes, transfers}}}}
    Every gate to every stop in the SAME region.
    Used in Phase 3 of query-time stitching to move a passenger from the
    destination gate to their final alighting stop.

connecting_chunks.pkl
    {entry_gate: {time_bucket: {exit_gate: {travel_time_minutes, transfers}}}}
    Every gate to every OTHER gate in the SAME region.
    Used when a journey passes THROUGH a region without starting or ending
    there — i.e. the passenger enters via one gate and exits via another
    without any inter-region hop in between.

gate_pairs.pkl
    {src_gate: {time_bucket: {dst_gate: {travel_time_minutes, trip_id}}}}
    Consecutive gate pairs on the same trip where src_gate and dst_gate are
    in DIFFERENT regions. These are the inter-region connections traversed
    by the gate Dijkstra at query time.
    time_bucket = nearest 15-min bucket of departure time at src_gate.
    travel_time = arrival_time(dst_gate) - arrival_time(src_gate) in minutes.
    Only the best (shortest) travel time per (src_gate, bucket, dst_gate) is
    kept — one entry per gate pair per time bucket.

intra_chunks.pkl
    {stop_id: {time_bucket: {dest_stop_id: {travel_time_minutes, transfers}}}}
    Non-gate stops to every reachable stop in the SAME region, computed
    using only that region's TBTR/RAPTOR dict.

    Purpose: covers direct intra-region journeys that never touch a gate.
    Without this chunk, a passenger travelling entirely within one region
    (e.g. two stops served by the same local bus that never crosses a
    boundary) would be invisible to the stitcher, because the gate-based
    path (ingress → Dijkstra → egress) requires at least one gate to be
    reachable from the origin stop.

    Why non-gate origins only: gate stops are already covered by the
    ingress → egress path. Adding them here would duplicate work and
    inflate the chunk size without adding coverage.

    At query time (hierarchical_query.py Phase 0) the intra results are
    computed first and then MERGED with the gate-based results, keeping
    the lower travel time per destination stop when both paths find the
    same destination.

All time buckets are 15-minute resolution across the full day (96 buckets).
Chunks are precomputed with a generous MAX_TRANSFER; the global transfer
budget is enforced at query time by HierarchicalStitcher.
"""

import os
import sys
import pickle
from collections import defaultdict
from time import time as now

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from miscellaneous_func import read_testcase
from Algorithms.TBTR.one_many_tbtr   import onetomany_rtbtr
from Algorithms.RAPTOR.one_many_raptor import onetomany_raptor


# ---------------------------------------------------------------------------
# Time bucket helpers
# ---------------------------------------------------------------------------

def generate_time_buckets(base_date: str):
    """96 pd.Timestamps at 15-min intervals for a full day."""
    start = pd.Timestamp(f"{base_date} 00:00:00")
    return [start + pd.Timedelta(minutes=15 * i) for i in range(96)]


def bucket_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%H:%M")


def nearest_bucket_label(ts, time_buckets) -> str:
    """Round a timestamp to the nearest 15-min bucket label."""
    ts = pd.Timestamp(ts)
    diffs = [abs((ts - b).total_seconds()) for b in time_buckets]
    return bucket_label(time_buckets[diffs.index(min(diffs))])


# ---------------------------------------------------------------------------
# TBTR loaders
# ---------------------------------------------------------------------------

def _load_region_tbtr(NETWORK_NAME, region_id):
    with open(f"./Data/TBTR/{NETWORK_NAME}/region_{region_id}.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Chunk computation (ingress / egress / connecting)
# ---------------------------------------------------------------------------

def _compute_chunks(source_list, destination_list, time_buckets,
                    tbtr_dict, routes_by_stop_dict, stops_dict,
                    stoptimes_dict, footpath_dict, idx_by_route_stop_dict,
                    MAX_TRANSFER, WALKING_FROM_SOURCE, desc="",
                    ENGINE="TBTR",
                    MAX_TOTAL_TIME_MIN=90, MAX_WALK_TIME_MIN=15):
    """
    For each source run the chosen engine once per time bucket.

    Both engines return the same format (int dest_id keys, travel_time_minutes
    and transfers fields), so no post-processing is needed here.

    ENGINE:
        "TBTR"   — uses onetomany_rtbtr; requires tbtr_dict (region pkl).
        "RAPTOR" — uses onetomany_raptor; tbtr_dict is ignored (pass None).

    Returns:
        {source_id: {bucket_label: {dest_id: {travel_time_minutes, transfers}}}}
    """
    result = {}
    for source_id in tqdm(source_list, desc=desc, leave=False):
        result[source_id] = {}
        for bucket_ts in time_buckets:
            lbl = bucket_label(bucket_ts)

            if ENGINE == "RAPTOR":
                bucket_result = onetomany_raptor(
                    SOURCE                 = source_id,
                    DESTINATION_LIST       = destination_list,
                    filter_time            = bucket_ts,
                    d_time_groups          = None,
                    MAX_TRANSFER           = MAX_TRANSFER,
                    WALKING_FROM_SOURCE    = WALKING_FROM_SOURCE,
                    PRINT_ITINERARY        = 0,
                    OPTIMIZED              = 0,
                    routes_by_stop_dict    = routes_by_stop_dict,
                    stops_dict             = stops_dict,
                    stoptimes_dict         = stoptimes_dict,
                    footpath_dict          = footpath_dict,
                    idx_by_route_stop_dict = idx_by_route_stop_dict,
                    trip_transfer_dict     = None,
                    trip_set               = set(),
                    MAX_TOTAL_TIME_MIN     = MAX_TOTAL_TIME_MIN,
                    MAX_WALK_TIME_MIN      = MAX_WALK_TIME_MIN,
                )
            else:  # TBTR (default)
                bucket_result = onetomany_rtbtr(
                    SOURCE                 = source_id,
                    DESTINATION_LIST       = destination_list,
                    source_departures      = [bucket_ts],
                    MAX_TRANSFER           = MAX_TRANSFER,
                    WALKING_FROM_SOURCE    = WALKING_FROM_SOURCE,
                    routes_by_stop_dict    = routes_by_stop_dict,
                    stops_dict             = stops_dict,
                    stoptimes_dict         = stoptimes_dict,
                    footpath_dict          = footpath_dict,
                    idx_by_route_stop_dict = idx_by_route_stop_dict,
                    trip_transfer_dict     = tbtr_dict,
                )

            result[source_id][lbl] = bucket_result
    return result


# ---------------------------------------------------------------------------
# Gate pairs computation (inter-region, consecutive gates on same trip)
# ---------------------------------------------------------------------------

def compute_gate_pairs(stoptimes_dict, gate_to_region, all_gates, time_buckets):
    """
    Scan every trip in stoptimes_dict for consecutive gate stops that belong
    to DIFFERENT regions. For each such pair record the best travel time
    per time bucket (bucket based on departure time at src_gate).

    Returns:
        {src_gate: {time_bucket: {dst_gate: {travel_time_minutes, trip_id}}}}
    """
    # gate_pairs_raw[src_gate][bucket][dst_gate] = best travel_time_minutes
    gate_pairs_raw = defaultdict(lambda: defaultdict(dict))

    print("  Scanning trips for consecutive inter-region gate pairs...")

    for route_id, trips in tqdm(stoptimes_dict.items(), desc="  gate_pairs", leave=False):
        for trip_idx, trip in enumerate(trips):
            trip_id = f"{route_id}_{trip_idx}"

            # Find all gate stops in this trip in order
            gate_stops = [
                (s_idx, stop_id, arr_time)
                for s_idx, (stop_id, arr_time) in enumerate(trip)
                if stop_id in all_gates
            ]

            if len(gate_stops) < 2:
                continue

            # Consecutive gate pairs only
            for i in range(len(gate_stops) - 1):
                src_idx,  src_gate,  src_arr  = gate_stops[i]
                dst_idx,  dst_gate,  dst_arr  = gate_stops[i + 1]

                # Must be in different regions for inter-region connection
                src_region = gate_to_region.get(src_gate)
                dst_region = gate_to_region.get(dst_gate)
                if src_region is None or dst_region is None:
                    continue
                if src_region == dst_region:
                    continue

                travel_min = (pd.Timestamp(dst_arr) -
                              pd.Timestamp(src_arr)).total_seconds() / 60
                if travel_min <= 0:
                    continue

                # Time bucket based on departure at src_gate
                bucket = nearest_bucket_label(src_arr, time_buckets)

                existing = gate_pairs_raw[src_gate][bucket].get(dst_gate)
                if (existing is None or
                        travel_min < existing["travel_time_minutes"]):
                    gate_pairs_raw[src_gate][bucket][dst_gate] = {
                        "travel_time_minutes": round(travel_min, 2),
                        "trip_id":             trip_id,
                    }

    # Convert nested defaultdicts to plain dicts
    gate_pairs = {
        src: {
            bucket: dict(dsts)
            for bucket, dsts in buckets.items()
        }
        for src, buckets in gate_pairs_raw.items()
    }

    # Stats
    total_pairs = sum(
        len(dsts)
        for buckets in gate_pairs.values()
        for dsts in buckets.values()
    )
    print(f"  Found {total_pairs} (src_gate, bucket, dst_gate) entries "
          f"across {len(gate_pairs)} source gates.")

    return gate_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def precompute_all_chunks(NETWORK_NAME, base_date,
                          MAX_TRANSFER=2, WALKING_FROM_SOURCE=1,
                          ENGINE="TBTR",
                          MAX_TOTAL_TIME_MIN=90, MAX_WALK_TIME_MIN=15):
    h_path = f"./Data/Hierarchical/{NETWORK_NAME}"

    print("Loading hierarchical partitions...")
    with open(f"{h_path}/regions.pkl", "rb") as f:
        regions = pickle.load(f)
    with open(f"{h_path}/gates.pkl", "rb") as f:
        gates_by_region = pickle.load(f)

    stops_by_region = defaultdict(list)
    for stop_id, region_id in regions.items():
        stops_by_region[region_id].append(stop_id)

    # Gate lookup structures
    gate_to_region = {
        g: r for r, gs in gates_by_region.items() for g in gs
    }
    all_gates = set(gate_to_region.keys())

    all_region_ids = sorted(stops_by_region.keys())

    print("Loading GTFS dictionaries...")
    (_, _, _, _,
     stops_dict, stoptimes_dict, footpath_dict,
     routes_by_stop_dict, idx_by_route_stop_dict, _) = read_testcase(NETWORK_NAME)

    time_buckets = generate_time_buckets(base_date)
    print(f"Time buckets: {len(time_buckets)} "
          f"({bucket_label(time_buckets[0])} to {bucket_label(time_buckets[-1])})")

    ingress_chunks    = {}
    egress_chunks     = {}
    connecting_chunks = {}
    intra_chunks      = {}

    t_total = now()

    # ==========================================================
    # Ingress, egress, connecting — one region at a time
    # ==========================================================
    for region_id in all_region_ids:
        r_stops = stops_by_region[region_id]
        r_gates = gates_by_region.get(region_id, [])

        if not r_gates:
            print(f"\nRegion {region_id}: no gates, skipping.")
            continue

        print(f"\nRegion {region_id}: {len(r_stops)} stops, {len(r_gates)} gates")

        try:
            region_tbtr = _load_region_tbtr(NETWORK_NAME, region_id)
        except FileNotFoundError:
            print(f"  WARNING: region_{region_id}.pkl not found, skipping.")
            continue

        # Ingress: every stop -> every gate (same region)
        print(f"  Ingress ({len(r_stops)} stops x {len(r_gates)} gates x 96 buckets)...")
        ingress_chunks.update(_compute_chunks(
            source_list=r_stops, destination_list=r_gates,
            time_buckets=time_buckets, tbtr_dict=region_tbtr,
            routes_by_stop_dict=routes_by_stop_dict, stops_dict=stops_dict,
            stoptimes_dict=stoptimes_dict, footpath_dict=footpath_dict,
            idx_by_route_stop_dict=idx_by_route_stop_dict,
            MAX_TRANSFER=MAX_TRANSFER, WALKING_FROM_SOURCE=WALKING_FROM_SOURCE,
            desc=f"  R{region_id} ingress",
            ENGINE=ENGINE,
            MAX_TOTAL_TIME_MIN=MAX_TOTAL_TIME_MIN,
            MAX_WALK_TIME_MIN=MAX_WALK_TIME_MIN,
        ))

        # Egress: every gate -> every stop (same region)
        print(f"  Egress ({len(r_gates)} gates x {len(r_stops)} stops x 96 buckets)...")
        egress_chunks.update(_compute_chunks(
            source_list=r_gates, destination_list=r_stops,
            time_buckets=time_buckets, tbtr_dict=region_tbtr,
            routes_by_stop_dict=routes_by_stop_dict, stops_dict=stops_dict,
            stoptimes_dict=stoptimes_dict, footpath_dict=footpath_dict,
            idx_by_route_stop_dict=idx_by_route_stop_dict,
            MAX_TRANSFER=MAX_TRANSFER, WALKING_FROM_SOURCE=WALKING_FROM_SOURCE,
            desc=f"  R{region_id} egress",
            ENGINE=ENGINE,
            MAX_TOTAL_TIME_MIN=MAX_TOTAL_TIME_MIN,
            MAX_WALK_TIME_MIN=MAX_WALK_TIME_MIN,
        ))

        # Connecting: every gate -> every OTHER gate (same region)
        if len(r_gates) > 1:
            print(f"  Connecting ({len(r_gates)} gates x {len(r_gates)-1} x 96 buckets)...")
            for entry_gate in r_gates:
                other_gates = [g for g in r_gates if g != entry_gate]
                connecting_chunks.update(_compute_chunks(
                    source_list=[entry_gate], destination_list=other_gates,
                    time_buckets=time_buckets, tbtr_dict=region_tbtr,
                    routes_by_stop_dict=routes_by_stop_dict, stops_dict=stops_dict,
                    stoptimes_dict=stoptimes_dict, footpath_dict=footpath_dict,
                    idx_by_route_stop_dict=idx_by_route_stop_dict,
                    MAX_TRANSFER=MAX_TRANSFER, WALKING_FROM_SOURCE=WALKING_FROM_SOURCE,
                    desc=f"  R{region_id} connecting gate {entry_gate}",
                    ENGINE=ENGINE,
                    MAX_TOTAL_TIME_MIN=MAX_TOTAL_TIME_MIN,
                    MAX_WALK_TIME_MIN=MAX_WALK_TIME_MIN,
                ))

        # Intra: every non-gate stop -> every stop in same region
        # Gate stops are excluded as origins — covered by ingress chunks
        non_gate_stops = [s for s in r_stops if s not in all_gates]
        if non_gate_stops:
            print(f"  Intra ({len(non_gate_stops)} non-gate stops x "
                  f"{len(r_stops)} dest stops x 96 buckets)...")
            intra_chunks.update(_compute_chunks(
                source_list=non_gate_stops, destination_list=r_stops,
                time_buckets=time_buckets, tbtr_dict=region_tbtr,
                routes_by_stop_dict=routes_by_stop_dict, stops_dict=stops_dict,
                stoptimes_dict=stoptimes_dict, footpath_dict=footpath_dict,
                idx_by_route_stop_dict=idx_by_route_stop_dict,
                MAX_TRANSFER=MAX_TRANSFER, WALKING_FROM_SOURCE=WALKING_FROM_SOURCE,
                desc=f"  R{region_id} intra",
                ENGINE=ENGINE,
                MAX_TOTAL_TIME_MIN=MAX_TOTAL_TIME_MIN,
                MAX_WALK_TIME_MIN=MAX_WALK_TIME_MIN,
            ))

        del region_tbtr
        print(f"  Region {region_id} done.")

    # ==========================================================
    # Gate pairs — inter-region consecutive gate connections
    # ==========================================================
    print("\nComputing gate pairs (inter-region consecutive gates on same trip)...")
    gate_pairs = compute_gate_pairs(
        stoptimes_dict, gate_to_region, all_gates, time_buckets)

    # ==========================================================
    # Save
    # ==========================================================
    print("\nSaving...")

    with open(f"{h_path}/ingress_chunks.pkl", "wb") as f:
        pickle.dump(ingress_chunks, f)
    print(f"  ingress_chunks.pkl    — {len(ingress_chunks)} stops")

    with open(f"{h_path}/egress_chunks.pkl", "wb") as f:
        pickle.dump(egress_chunks, f)
    print(f"  egress_chunks.pkl     — {len(egress_chunks)} gates")

    with open(f"{h_path}/connecting_chunks.pkl", "wb") as f:
        pickle.dump(connecting_chunks, f)
    print(f"  connecting_chunks.pkl — {len(connecting_chunks)} entry gates")

    with open(f"{h_path}/gate_pairs.pkl", "wb") as f:
        pickle.dump(gate_pairs, f)
    print(f"  gate_pairs.pkl        — {len(gate_pairs)} source gates")

    with open(f"{h_path}/intra_chunks.pkl", "wb") as f:
        pickle.dump(intra_chunks, f)
    print(f"  intra_chunks.pkl      — {len(intra_chunks)} non-gate stops")

    # --- Save JSON copies for inspection ---
    import json

    def _to_json_keys(d):
        """
        Recursively convert all dict keys to strings so JSON can serialize
        integer keys (JSON only allows string keys).
        """
        if isinstance(d, dict):
            return {str(k): _to_json_keys(v) for k, v in d.items()}
        return d

    print("\nSaving JSON copies for inspection...")

    with open(f"{h_path}/ingress_chunks.json", "w") as f:
        json.dump(_to_json_keys(ingress_chunks), f, indent=2)
    print(f"  ingress_chunks.json")

    with open(f"{h_path}/egress_chunks.json", "w") as f:
        json.dump(_to_json_keys(egress_chunks), f, indent=2)
    print(f"  egress_chunks.json")

    with open(f"{h_path}/connecting_chunks.json", "w") as f:
        json.dump(_to_json_keys(connecting_chunks), f, indent=2)
    print(f"  connecting_chunks.json")

    with open(f"{h_path}/gate_pairs.json", "w") as f:
        json.dump(_to_json_keys(gate_pairs), f, indent=2)
    print(f"  gate_pairs.json")

    with open(f"{h_path}/intra_chunks.json", "w") as f:
        json.dump(_to_json_keys(intra_chunks), f, indent=2)
    print(f"  intra_chunks.json")

    print(f"\nDone. Total time: {round((now() - t_total) / 60, 2)} mins")


if __name__ == "__main__":
    with open("./parameters_entered.txt", "rb") as f:
        parameter_files = pickle.load(f)
    NETWORK_NAME = parameter_files[1]

    default_date = "2025-01-01"
    base_date = input(
        f"Analysis date for time buckets "
        f"(YYYY-MM-DD, default: {default_date})\n: ").strip() or default_date

    MAX_TRANSFER        = int(input("Max transfers (default: 2)\n: ") or 2)
    WALKING_FROM_SOURCE = int(input("Allow walking from source? 1/0 (default: 1)\n: ") or 1)

    engine_choice = input(
        "Routing engine for chunk computation:\n"
        "  1 — TBTR  (uses precomputed trip-transfer dicts, more accurate)\n"
        "  2 — RAPTOR (faster, no trip-transfer dicts needed)\n"
        ": ").strip()
    ENGINE = "RAPTOR" if engine_choice == "2" else "TBTR"
    print(f"Engine: {ENGINE}")

    MAX_TOTAL_TIME_MIN = float(input("Max total journey time in minutes (default: 90)\n: ") or 90)
    MAX_WALK_TIME_MIN  = float(input("Max walking time in minutes (default: 15)\n: ") or 15)

    precompute_all_chunks(
        NETWORK_NAME, base_date,
        MAX_TRANSFER, WALKING_FROM_SOURCE,
        ENGINE=ENGINE,
        MAX_TOTAL_TIME_MIN=MAX_TOTAL_TIME_MIN,
        MAX_WALK_TIME_MIN=MAX_WALK_TIME_MIN,
    )