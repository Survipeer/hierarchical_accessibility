"""
Builds region-partitioned TBTR trip-transfer dictionaries.

Memory strategy — process ONE region at a time
-----------------------------------------------
For each region:
  1. Filter stoptimes/routes to only routes that have stops in this region
  2. Run Algorithm 1 on those routes only
  3. Run Algorithm 2 on those transfers only
  4. Run Algorithm 3 on those trips only
  5. Keep only transfers where the boarding stop belongs to THIS region
     (cross-boundary transfers go to gates_dict_raw instead)
  6. Save region_X.pkl and free memory immediately

Then for gates:
  Run the same three algorithms on cross-boundary routes only
  Save gates.pkl

Peak memory = one region's routes/transfers at a time, not the whole network.

Output files
------------
./Data/TBTR/{NETWORK_NAME}/region_{region_id}.pkl  — one per region
./Data/TBTR/{NETWORK_NAME}/gates.pkl               — cross-boundary transfers
"""

import multiprocessing
import sys
from collections import defaultdict
from itertools import chain
from multiprocessing import get_context
from random import shuffle
from time import time as time_measure
import os
import pickle
import gc
import pandas as pd
import platform
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from miscellaneous_func import *

# ---------------------------------------------------------------------------
# Algorithm 1 — enumerate candidate transfers
# (identical logic to original, boarding_stop included in output)
# ---------------------------------------------------------------------------
routes_by_stop_dict = None
stops_dict          = None
stoptimes_dict      = None
change_time         = None
footpath_dict       = None

def init_algorithm1(routes_by_stop_dict_, stops_dict_, stoptimes_dict_,
                    change_time_, footpath_dict_):
    global routes_by_stop_dict, stops_dict, stoptimes_dict, change_time, footpath_dict
    routes_by_stop_dict = routes_by_stop_dict_
    stops_dict          = stops_dict_
    stoptimes_dict      = stoptimes_dict_
    change_time         = change_time_
    footpath_dict       = footpath_dict_


def algorithm1_parallel(route_details: tuple) -> list:
    trip_transfer_list = []
    rr, route_trips = route_details

    for tcount, trips in enumerate(route_trips):
        for scount, stop_seq in enumerate(trips[1:], 1):

            # Direct board at same stop
            try:
                to_route_list = routes_by_stop_dict[stop_seq[0]].copy()
                to_route_list.remove(rr)
                for r_route in to_route_list:
                    stopindex_by_route = stops_dict[r_route].index(stop_seq[0])
                    earliest_trip = -1
                    for ttcount, tripss in enumerate(stoptimes_dict[r_route]):
                        if tripss[stopindex_by_route][1] >= change_time + stop_seq[1]:
                            earliest_trip = 1
                            break
                    if earliest_trip == 1:
                        if r_route != rr or tcount < ttcount or stopindex_by_route < scount:
                            trip_transfer_list.append(
                                (f"{rr}_{tcount}", scount,
                                 f"{r_route}_{ttcount}", stopindex_by_route,
                                 stop_seq[0]))
            except KeyError:
                pass

            # Board via footpath
            try:
                for connection in footpath_dict[stop_seq[0]]:
                    to_route_list = routes_by_stop_dict[connection[0]]
                    for r_route in to_route_list:
                        stopindex_by_route = stops_dict[r_route].index(connection[0])
                        earliest_trip = -1
                        for ttcount, tripss in enumerate(stoptimes_dict[r_route]):
                            if tripss[stopindex_by_route][1] >= stop_seq[1] + connection[1]:
                                earliest_trip = 1
                                break
                        if earliest_trip == 1:
                            if r_route != rr or tcount < ttcount or stopindex_by_route < scount:
                                trip_transfer_list.append(
                                    (f"{rr}_{tcount}", scount,
                                     f"{r_route}_{ttcount}", stopindex_by_route,
                                     connection[0]))
            except KeyError:
                pass

    return trip_transfer_list


# ---------------------------------------------------------------------------
# Algorithm 2 — remove U-turns
# ---------------------------------------------------------------------------
stoptimes_dict_algo2 = None

def init_algorithm2(stoptimes_dict_):
    global stoptimes_dict_algo2
    stoptimes_dict_algo2 = stoptimes_dict_


def algorithm2_parallel(trip_transfer_: list):
    try:
        from_stop_dep = stoptimes_dict_algo2[trip_transfer_[1]][trip_transfer_[2]][trip_transfer_[5]]
        to_stop_det   = stoptimes_dict_algo2[trip_transfer_[3]][trip_transfer_[4]][trip_transfer_[6]]
        if from_stop_dep[0] == to_stop_det[0]:
            if to_stop_det[1] <= from_stop_dep[1]:
                return trip_transfer_[0]
        return []
    except IndexError:
        return []


# ---------------------------------------------------------------------------
# Algorithm 3 — remove dominated transfers
# ---------------------------------------------------------------------------
footpath_dict_algo3      = None
trip_transfer_dict_algo3 = None
stoptimes_dict_algo3     = None
inf_time_algo3           = None
footpath_keys_algo3      = None

def init_algorithm3(footpath_dict_, trip_transfer_dict_, stoptimes_dict_,
                    inf_time_, footpath_keys_):
    global footpath_dict_algo3, trip_transfer_dict_algo3, stoptimes_dict_algo3
    global inf_time_algo3, footpath_keys_algo3
    footpath_dict_algo3      = footpath_dict_
    trip_transfer_dict_algo3 = trip_transfer_dict_
    stoptimes_dict_algo3     = stoptimes_dict_
    inf_time_algo3           = inf_time_
    footpath_keys_algo3      = footpath_keys_


def algorithm3_parallel(trip_details: tuple) -> list:
    r_id, t_id, trip = trip_details
    removed_trans = []
    stop_labels   = defaultdict(lambda: inf_time_algo3)
    trip_rev      = reversed(list(enumerate(trip)))
    tid           = f"{r_id}_{t_id}"

    for s_idx, stop_seq in trip_rev:
        stop_labels[stop_seq[0]] = min(stop_labels[stop_seq[0]], stop_seq[1])
        try:
            for q in footpath_dict_algo3[stop_seq[0]]:
                stop_labels[q[0]] = min(stop_labels[q[0]], stop_seq[1] + q[1])
        except KeyError:
            pass
        try:
            trans_from_stop = [
                (trans, [int(x) for x in trans[1].split("_")])
                for trans in trip_transfer_dict_algo3[tid]
                if trans[0] == s_idx
            ]
            for trans, breakdown in trans_from_stop:
                keep = False
                for stop_connect_0, stop_connect_1 in \
                        stoptimes_dict_algo3[breakdown[0]][breakdown[1]][trans[2] + 1:]:
                    if stop_connect_1 < stop_labels[stop_connect_0]:
                        keep = True
                        stop_labels[stop_connect_0] = stop_connect_1
                    if stop_connect_0 in footpath_keys_algo3:
                        for footpath_connect in footpath_dict_algo3[stop_connect_0]:
                            if stop_labels[footpath_connect[0]] > stop_connect_1 + footpath_connect[1]:
                                keep = True
                                stop_labels[footpath_connect[0]] = stop_connect_1 + footpath_connect[1]
                if not keep:
                    removed_trans.append((tid, trans))
        except KeyError:
            pass

    return removed_trans


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_final_dict(trip_transfer_dict, stops_dict, essential_trips):
    """
    Convert flat list-valued dict to nested format:
        { trip_id_str : { stop_index : [(to_trip_str, to_stop_idx), ...] } }
    """
    result = {}
    for tid, connections in trip_transfer_dict.items():
        if not connections and tid not in essential_trips:
            continue
        result[tid] = {}
        for from_stop_idx, to_trip, to_stop_idx in connections:
            result[tid].setdefault(from_stop_idx, []).append((to_trip, to_stop_idx))

    for tid in list(result.keys()):
        route_id = int(tid.split("_")[0])
        all_stop_indices = set(range(len(stops_dict[route_id])))
        for key in all_stop_indices - set(result[tid].keys()):
            result[tid][key] = []

    return result


def _save(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved {len(data)} trip entries -> {path}")


def _run_three_algorithms(route_details_list, stoptimes_dict_local,
                          stops_dict, routes_by_stop_dict, footpath_dict,
                          change_time, USE_PARALLEL, CORES, label=""):
    """
    Run Algorithms 1, 2, 3 on a subset of routes/trips.
    Returns trip_transfer_dict for that subset.
    stoptimes_dict_local: full stoptimes_dict (needed for algo1 to look up
                          connecting routes, even if they're outside the region).
    """
    # ---- Algorithm 1 ----
    shuffle(route_details_list)
    if USE_PARALLEL == 1:
        start_method = "fork" if platform.system() != "Windows" else "spawn"
        ctx = get_context(start_method)
        with ctx.Pool(CORES, initializer=init_algorithm1,
                      initargs=(routes_by_stop_dict, stops_dict,
                                stoptimes_dict_local, change_time,
                                footpath_dict)) as pool:
            result = list(tqdm(
                pool.imap_unordered(algorithm1_parallel, route_details_list),
                total=len(route_details_list), desc=f"  Algo1{label}"))
    else:
        init_algorithm1(routes_by_stop_dict, stops_dict,
                        stoptimes_dict_local, change_time, footpath_dict)
        result = [algorithm1_parallel(rd)
                  for rd in tqdm(route_details_list, desc=f"  Algo1{label}")]

    result = list(chain.from_iterable(result))
    result = [(idx, fr, fr_idx, to, to_idx, bstop)
              for idx, (fr, fr_idx, to, to_idx, bstop) in enumerate(result)]
    print(f"  Candidate transfers: {len(result)}")

    if not result:
        return defaultdict(list)

    # ---- Algorithm 2 ----
    processed_result = [
        (idx, fr_rid, fr_tid, to_rid, to_tid, fs_idx - 1, ts_idx + 1)
        for (idx, fr, fs_idx, to, ts_idx, _bstop) in result
        for fr_rid, fr_tid in [map(int, fr.split("_"))]
        for to_rid, to_tid in [map(int, to.split("_"))]
    ]
    if USE_PARALLEL == 1:
        start_method = "fork" if platform.system() != "Windows" else "spawn"
        ctx = get_context(start_method)
        with ctx.Pool(CORES, initializer=init_algorithm2,
                      initargs=(stoptimes_dict_local,)) as pool:
            U_Turns_list = list(tqdm(
                pool.imap_unordered(algorithm2_parallel, processed_result),
                total=len(processed_result), desc=f"  Algo2{label}"))
    else:
        init_algorithm2(stoptimes_dict_local)
        U_Turns_list = [algorithm2_parallel(t)
                        for t in tqdm(processed_result, desc=f"  Algo2{label}")]

    U_Turns_set  = set(x for x in U_Turns_list if x)
    Transfer_set = [x for x in result if x[0] not in U_Turns_set]
    print(f"  After U-turn removal: {len(Transfer_set)}")
    del result, processed_result, U_Turns_list

    trip_transfer_dict = defaultdict(list)
    for idx, from_trip, from_stop_idx, to_trip, to_stop_idx, _bstop in Transfer_set:
        trip_transfer_dict[from_trip].append((from_stop_idx, to_trip, to_stop_idx))
    del Transfer_set

    # ---- Algorithm 3 ----
    footpath_dict_seconds = {
        stop: [(y[0], y[1].total_seconds()) for y in flist]
        for stop, flist in footpath_dict.items()
    }
    stoptimes_dict_ts = {
        rid: [[(s[0], s[1].timestamp()) for s in trip] for trip in trips]
        for rid, trips in stoptimes_dict_local.items()
    }
    inf_time      = (pd.to_datetime("today").round(freq="h") +
                     pd.to_timedelta("365 day")).timestamp()
    footpath_keys = set(footpath_dict_seconds.keys())

    trip_list = [
        (rid, t_id, trip)
        for rid, route_trips in sorted(stoptimes_dict_ts.items())
        for t_id, trip in enumerate(route_trips)
    ]

    if USE_PARALLEL == 1:
        start_method = "fork" if platform.system() != "Windows" else "spawn"
        ctx = get_context(start_method)
        with ctx.Pool(CORES, initializer=init_algorithm3,
                      initargs=(footpath_dict_seconds, trip_transfer_dict,
                                stoptimes_dict_ts, inf_time, footpath_keys)) as pool:
            non_optimal_trans = list(tqdm(
                pool.imap_unordered(algorithm3_parallel, trip_list),
                total=len(trip_list), desc=f"  Algo3{label}"))
    else:
        init_algorithm3(footpath_dict_seconds, trip_transfer_dict,
                        stoptimes_dict_ts, inf_time, footpath_keys)
        non_optimal_trans = [algorithm3_parallel(t)
                             for t in tqdm(trip_list, desc=f"  Algo3{label}")]

    for route_level_turns in non_optimal_trans:
        for tid, trans in route_level_turns:
            if trans in trip_transfer_dict[tid]:
                trip_transfer_dict[tid].remove(trans)

    total = sum(len(v) for v in trip_transfer_dict.values())
    print(f"  After dominated-transfer removal: {total}")
    del stoptimes_dict_ts, footpath_dict_seconds, non_optimal_trans, trip_list
    gc.collect()

    return trip_transfer_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    with open("./parameters_entered.txt", "rb") as file:
        parameter_files = pickle.load(file)
    BUILD_TRANSFER, NETWORK_NAME, BUILD_TBTR_FILES, \
        BUILD_TRANSFER_PATTERNS_FILES, BUILD_CSA = parameter_files

    if BUILD_TBTR_FILES != 1:
        print("BUILD_TBTR_FILES is 0 — nothing to do.")
        sys.exit(0)

    breaker = "________________________________"
    print("Building region-partitioned TBTR trip-transfer dicts.\n")

    USE_PARALLEL = int(input(
        "TBTR can be built in parallel. Enter 1 to use multiprocessing. "
        "Else press 0.\n: "))
    CORES = 0
    if USE_PARALLEL != 0:
        CORES = int(input(
            f"Enter number of CORES (>=1). "
            f"Available: {multiprocessing.cpu_count()}\n: "))

    change_time = pd.to_timedelta(0, unit="seconds")

    GENERATE_LOGFILE = int(input(
        "Press 1 to redirect output to a log file. Else press 0.\n: "))

    os.makedirs("./logs", exist_ok=True)
    os.makedirs(f"./Data/TBTR/{NETWORK_NAME}", exist_ok=True)

    # ------------------------------------------------------------------
    # Load GTFS dicts (read-only — never modified)
    # ------------------------------------------------------------------
    print("Loading GTFS data...")
    (stops_file, trips_file, stop_times_file, transfers_file,
     stops_dict, stoptimes_dict, footpath_dict,
     routes_by_stop_dict, idx_by_route_stop_dict,
     routesindx_by_stop_dict) = read_testcase(NETWORK_NAME)

    # ------------------------------------------------------------------
    # Load region metadata
    # ------------------------------------------------------------------
    h_path = f"./Data/Hierarchical/{NETWORK_NAME}"
    try:
        with open(f"{h_path}/regions.pkl", "rb") as f:
            regions = pickle.load(f)         # {stop_id: region_id}
        with open(f"{h_path}/gates.pkl", "rb") as f:
            gates_by_region = pickle.load(f) # {region_id: [gate_stop_id,...]}
        print(f"Loaded {len(set(regions.values()))} regions.")
    except FileNotFoundError:
        print("ERROR: regions.pkl / gates.pkl not found.")
        print("Run hierarchical_setup.py first.")
        sys.exit(1)

    stop_to_region = dict(regions)
    all_gate_stops = set(s for gs in gates_by_region.values() for s in gs)
    all_region_ids = sorted(set(regions.values()))

    # Pre-compute which regions each route touches
    route_to_regions = {}
    for rid in stoptimes_dict:
        r_regions = set()
        for s in stops_dict[rid]:
            r = stop_to_region.get(s, -1)
            if r != -1:
                r_regions.add(r)
        route_to_regions[rid] = r_regions

    # Group routes by region (a route appears in every region it touches)
    routes_in_region = defaultdict(set)
    for rid, r_set in route_to_regions.items():
        for r in r_set:
            routes_in_region[r].add(rid)

    # Cross-boundary routes: routes that touch at least two different regions
    cross_boundary_routes = {
        rid for rid, r_set in route_to_regions.items() if len(r_set) > 1
    }
    print(f"Cross-boundary routes (for gates dict): {len(cross_boundary_routes)}")

    # Essential trips: first + last trip of every route
    essential_trips = set()
    for rid, route_trips in stoptimes_dict.items():
        if route_trips:
            essential_trips.add(f"{rid}_0")
            essential_trips.add(f"{rid}_{len(route_trips) - 1}")

    if GENERATE_LOGFILE == 1:
        sys.stdout = open(f"./logs/tbtr_builder_{NETWORK_NAME}", "w")

    print(f"\nNetwork : {NETWORK_NAME}")
    print(f"CORES   : {CORES}")
    print(breaker)

    t0 = time_measure()
    gates_dict_raw = defaultdict(list)  # accumulates cross-boundary transfers

    # ==================================================================
    # Process each region independently
    # ==================================================================
    for region_id in all_region_ids:
        region_routes = routes_in_region[region_id]
        if not region_routes:
            print(f"\nRegion {region_id}: no routes, skipping.")
            continue

        print(f"\n{'='*40}")
        print(f"Region {region_id}  ({len(region_routes)} routes)")
        print(f"{'='*40}")

        # Build a LOCAL stoptimes_dict containing only this region's routes
        # PLUS any routes those routes can transfer TO (needed for algo1
        # to find connecting trips at transfer stops)
        # For memory efficiency we include only routes reachable in 1 transfer
        transfer_reachable = set()
        for rid in region_routes:
            for s in stops_dict[rid]:
                for r2 in routes_by_stop_dict.get(s, []):
                    transfer_reachable.add(r2)
                if s in footpath_dict:
                    for nb_stop, _ in footpath_dict[s]:
                        for r2 in routes_by_stop_dict.get(nb_stop, []):
                            transfer_reachable.add(r2)

        routes_needed = region_routes | transfer_reachable
        stoptimes_local = {rid: stoptimes_dict[rid]
                           for rid in routes_needed
                           if rid in stoptimes_dict}

        # route_details_list: only THIS region's routes go through algo1
        # (transfer_reachable routes are available for lookup but not iterated)
        route_details_list = [(rid, stoptimes_dict[rid])
                              for rid in region_routes
                              if rid in stoptimes_dict]

        # Run three algorithms on this region's routes
        region_ttd = _run_three_algorithms(
            route_details_list = route_details_list,
            stoptimes_dict_local = stoptimes_local,
            stops_dict           = stops_dict,
            routes_by_stop_dict  = routes_by_stop_dict,
            footpath_dict        = footpath_dict,
            change_time          = change_time,
            USE_PARALLEL         = USE_PARALLEL,
            CORES                = CORES,
            label                = f" R{region_id}",
        )

        # Add essential trips for this region
        for trip_key in essential_trips:
            rid = int(trip_key.split("_")[0])
            if region_id in route_to_regions.get(rid, set()):
                region_ttd.setdefault(trip_key, [])

        # ----------------------------------------------------------
        # Partition: keep only transfers where boarding stop is in
        # this region; cross-boundary ones go to gates_dict_raw too
        # ----------------------------------------------------------
        region_final_raw = defaultdict(list)

        for tid, connections in region_ttd.items():
            route_id = int(tid.split("_")[0])
            trip_idx = int(tid.split("_")[1])
            trip = stoptimes_dict.get(route_id, [[]])[trip_idx] \
                   if trip_idx < len(stoptimes_dict.get(route_id, [])) else []

            for from_stop_idx, to_trip, to_stop_idx in connections:
                if from_stop_idx >= len(trip):
                    continue
                boarding_stop = trip[from_stop_idx][0]
                src_region    = stop_to_region.get(boarding_stop, -1)

                # Only keep transfers boarding in THIS region
                if src_region != region_id:
                    continue

                to_route_id = int(to_trip.split("_")[0])
                to_stop     = stops_dict[to_route_id][to_stop_idx]
                dst_region  = stop_to_region.get(to_stop, -1)

                entry = (from_stop_idx, to_trip, to_stop_idx)
                region_final_raw[tid].append(entry)

                # Cross-boundary: also add to gates accumulator
                if dst_region != region_id and dst_region != -1:
                    gates_dict_raw[tid].append(entry)

            # Ensure essential trips present
            if tid in essential_trips:
                region_final_raw.setdefault(tid, [])

        # Build final nested format and save immediately
        final = _build_final_dict(region_final_raw, stops_dict, essential_trips)
        out_path = f"./Data/TBTR/{NETWORK_NAME}/region_{region_id}.pkl"
        _save(final, out_path)

        # Free memory before next region
        del region_ttd, region_final_raw, final
        del stoptimes_local, route_details_list, routes_needed, transfer_reachable
        gc.collect()
        print(f"  Region {region_id} done.")

    # ==================================================================
    # Gates dict — cross-boundary transfers collected above
    # Run Algorithm 3 pruning on the gates dict for extra quality
    # (Algos 1 & 2 were already applied per-region above)
    # ==================================================================
    print(f"\n{'='*40}")
    print(f"Gates dict  ({len(gates_dict_raw)} trips with cross-boundary transfers)")
    print(f"{'='*40}")

    # Identify which routes are involved in cross-boundary transfers
    gate_route_ids = set(int(tid.split("_")[0]) for tid in gates_dict_raw)
    gate_route_ids |= set(
        int(to_trip.split("_")[0])
        for conns in gates_dict_raw.values()
        for _, to_trip, _ in conns
    )

    stoptimes_gates = {rid: stoptimes_dict[rid]
                       for rid in gate_route_ids
                       if rid in stoptimes_dict}

    # Run Algorithm 3 pruning on the gates dict
    footpath_dict_seconds = {
        stop: [(y[0], y[1].total_seconds()) for y in flist]
        for stop, flist in footpath_dict.items()
    }
    stoptimes_gates_ts = {
        rid: [[(s[0], s[1].timestamp()) for s in trip] for trip in trips]
        for rid, trips in stoptimes_gates.items()
    }
    inf_time      = (pd.to_datetime("today").round(freq="h") +
                     pd.to_timedelta("365 day")).timestamp()
    footpath_keys = set(footpath_dict_seconds.keys())

    trip_list_gates = [
        (rid, t_id, trip)
        for rid, route_trips in sorted(stoptimes_gates_ts.items())
        for t_id, trip in enumerate(route_trips)
        if f"{rid}_{t_id}" in gates_dict_raw
    ]

    print(f"  Running Algorithm 3 on {len(trip_list_gates)} gate trips...")
    if USE_PARALLEL == 1:
        start_method = "fork" if platform.system() != "Windows" else "spawn"
        ctx = get_context(start_method)
        with ctx.Pool(CORES, initializer=init_algorithm3,
                      initargs=(footpath_dict_seconds, gates_dict_raw,
                                stoptimes_gates_ts, inf_time, footpath_keys)) as pool:
            non_optimal_gates = list(tqdm(
                pool.imap_unordered(algorithm3_parallel, trip_list_gates),
                total=len(trip_list_gates), desc="  Algo3 gates"))
    else:
        init_algorithm3(footpath_dict_seconds, gates_dict_raw,
                        stoptimes_gates_ts, inf_time, footpath_keys)
        non_optimal_gates = [algorithm3_parallel(t)
                             for t in tqdm(trip_list_gates, desc="  Algo3 gates")]

    for route_level_turns in non_optimal_gates:
        for tid, trans in route_level_turns:
            if trans in gates_dict_raw[tid]:
                gates_dict_raw[tid].remove(trans)

    total_gates = sum(len(v) for v in gates_dict_raw.values())
    print(f"  After pruning: {total_gates} gate transfer entries")

    # Add essential trips
    for trip_key in essential_trips:
        rid = int(trip_key.split("_")[0])
        if rid in gate_route_ids:
            gates_dict_raw.setdefault(trip_key, [])

    gates_final = _build_final_dict(dict(gates_dict_raw), stops_dict, essential_trips)
    gates_path  = f"./Data/TBTR/{NETWORK_NAME}/gates.pkl"
    _save(gates_final, gates_path)

    print(breaker)
    print(f"DONE.  Total time = {round((time_measure() - t0) / 60, 2)} mins")
    print(f"Saved {len(all_region_ids)} region files + 1 gates file "
          f"in ./Data/TBTR/{NETWORK_NAME}/")

    if GENERATE_LOGFILE == 1:
        sys.stdout.close()