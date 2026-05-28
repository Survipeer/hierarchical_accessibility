"""
One-To-Many rTBTR for Hierarchical Routing.

Key design decisions vs the old version:
-----------------------------------------
1. PROFILE query (no filter_time).
   For chunk precomputation we want the best travel time from a source stop
   to each destination across the *whole day*, not from a single departure.
   We iterate over every departure from the source (one per hour, sampled)
   and keep the best result per destination.

2. Uses trip_transfer_dict.
   The old version ignored trip_transfer_dict entirely, which meant the
   TBTR rounds were doing nothing useful — no transfers were ever found.
   We now look up precomputed transfers exactly like the original query code.

3. Clean return format.
   Returns {dest_id: {'travel_time_minutes': float, 'transfers': int,
                       'last_trip_id': str}}
   travel_time_minutes is the best journey time found across all departures.
   This is what regional_precalc and hierarchical_query both need.

4. No pandas groupby dependency.
   Departures are passed in as a plain list of (trip_id_str, arrival_time)
   tuples, already filtered to the source stop. This avoids keeping the
   full stop_times_file in memory inside every worker.
"""

import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithms.TBTR.TBTR_functions import (
    initialize_onemany,
    initialize_from_desti_onemany,
)


def _enqueue_trip(trip_id_str, board_stop_idx, R_t, Q, round_no,
                  stoptimes_dict, MAX_TRANSFER):
    """
    Add a trip to round `round_no` of Q if it improves R_t.
    Propagates the stop-index bound forward to later trips on the same route.
    """
    if board_stop_idx >= R_t[round_no][trip_id_str]:
        return
    route_id, tid_idx = (int(x) for x in trip_id_str.split("_"))
    Q[round_no].append((board_stop_idx, trip_id_str, route_id, tid_idx))
    # All later trips on the same route can also board here
    for later_idx in range(tid_idx, len(stoptimes_dict[route_id])):
        later_str = f"{route_id}_{later_idx}"
        if R_t[round_no][later_str] > board_stop_idx:
            R_t[round_no][later_str] = board_stop_idx


def _initialize_from_source(SOURCE, departure_time, WALKING_FROM_SOURCE,
                             routes_by_stop_dict, idx_by_route_stop_dict,
                             stoptimes_dict, footpath_dict, MAX_TRANSFER):
    """
    Find the earliest boardable trip from SOURCE (and walkable neighbours)
    at or after departure_time.  Returns (R_t, Q).
    """
    R_t = {r: defaultdict(lambda: 1000) for r in range(MAX_TRANSFER + 2)}
    Q   = [[] for _ in range(MAX_TRANSFER + 2)]

    candidate_stops = [(SOURCE, pd.to_timedelta(0, unit="s"))]
    if WALKING_FROM_SOURCE and SOURCE in footpath_dict:
        for nb_stop, walk_time in footpath_dict[SOURCE]:
            candidate_stops.append((nb_stop, walk_time))

    for stop, walk_time in candidate_stops:
        earliest_dep = departure_time + walk_time
        if stop not in routes_by_stop_dict:
            continue
        for route_id in routes_by_stop_dict[stop]:
            stop_idx = idx_by_route_stop_dict.get((route_id, stop))
            if stop_idx is None:
                continue
            # Find the first trip on this route that we can board
            for trip_idx, trip in enumerate(stoptimes_dict[route_id]):
                if trip[stop_idx][1] >= earliest_dep:
                    _enqueue_trip(f"{route_id}_{trip_idx}", stop_idx,
                                  R_t, Q, 1, stoptimes_dict, MAX_TRANSFER)
                    break

    return R_t, Q


def onetomany_rtbtr(SOURCE,
                    DESTINATION_LIST,
                    source_departures,
                    MAX_TRANSFER,
                    WALKING_FROM_SOURCE,
                    routes_by_stop_dict,
                    stops_dict,
                    stoptimes_dict,
                    footpath_dict,
                    idx_by_route_stop_dict,
                    trip_transfer_dict):
    """
    Profile one-to-many rTBTR.

    Args:
        SOURCE (int): source stop id.
        DESTINATION_LIST (list[int]): destination stop ids.
        source_departures (list[pd.Timestamp]): departure times to evaluate.
            For chunk precomputation pass one timestamp per hour (or a
            representative sample).  For a single-departure query pass a
            list with one element.
        MAX_TRANSFER (int): maximum number of transfers allowed.
        WALKING_FROM_SOURCE (int): 1 to allow initial walk from SOURCE.
        routes_by_stop_dict (dict): {stop_id: [route_id, ...]}.
        stops_dict (dict): {route_id: [stop_id, ...]}.
        stoptimes_dict (dict): {route_id: [[(stop_id, arrival_time), ...], ...]}.
        footpath_dict (dict): {stop_id: [(to_stop_id, timedelta), ...]}.
        idx_by_route_stop_dict (dict): {(route_id, stop_id): stop_index}.
        trip_transfer_dict (dict): per-region or gates TBTR dict loaded from
            build_TBTR_dict output.  Format:
            {trip_id_str: {stop_index: [(to_trip_str, to_stop_idx), ...]}}.

    Returns:
        dict: {dest_id: {'travel_time_minutes': float,
                          'transfers': int,
                          'last_trip_id': str or None}}
        Only destinations that were reached are included.
    """
    if not DESTINATION_LIST:
        return {}

    inf_time = (pd.to_datetime("today").round(freq="h")
                + pd.to_timedelta("365 day"))

    # L[dest][route] = [(stop_idx_in_route, foot_time, alighting_stop), ...]
    L = initialize_from_desti_onemany(
        routes_by_stop_dict, stops_dict, DESTINATION_LIST,
        footpath_dict, idx_by_route_stop_dict)

    # Best result per destination across all departure times
    # best[dest] = (best_travel_timedelta, best_transfers, best_last_trip)
    best = {}

    for dep_time in source_departures:

        # J[dest][round] = [arrival_time, last_trip_id_str]
        J, _ = initialize_onemany(MAX_TRANSFER, DESTINATION_LIST)

        R_t, Q = _initialize_from_source(
            SOURCE, dep_time, WALKING_FROM_SOURCE,
            routes_by_stop_dict, idx_by_route_stop_dict,
            stoptimes_dict, footpath_dict, MAX_TRANSFER)

        for n in range(1, MAX_TRANSFER + 1):
            if not Q[n]:
                continue

            for board_stop_idx, trip_id_str, route_id, tid_idx in Q[n]:
                trip = stoptimes_dict[route_id][tid_idx]

                # --- Check if this trip reaches any destination ---
                if route_id in L:
                    # L is keyed by dest then by route
                    pass  # handled per-dest below

                for desti in DESTINATION_LIST:
                    if route_id not in L[desti]:
                        continue
                    for alight_stop_idx, foot_time, alight_stop in L[desti][route_id]:
                        if board_stop_idx >= alight_stop_idx:
                            continue
                        # Find alight_stop in the trip slice
                        for s_idx in range(board_stop_idx, len(trip)):
                            if trip[s_idx][0] == alight_stop:
                                arr = trip[s_idx][1] + foot_time
                                if arr < J[desti][n][0]:
                                    J[desti][n][0] = arr
                                    J[desti][n][1] = trip_id_str
                                    # Propagate to higher transfer rounds
                                    for r in range(n + 1, MAX_TRANSFER + 1):
                                        if J[desti][r][0] > arr:
                                            J[desti][r][0] = arr
                                            J[desti][r][1] = trip_id_str
                                break

                # --- Expand transfers from this trip ---
                if trip_id_str in trip_transfer_dict:
                    transfers_from_trip = trip_transfer_dict[trip_id_str]
                    # transfers_from_trip: {stop_idx: [(to_trip_str, to_stop_idx),...]}
                    for from_s_idx, connections in transfers_from_trip.items():
                        if from_s_idx <= board_stop_idx:
                            continue
                        if from_s_idx >= len(trip):
                            continue
                        for to_trip_str, to_stop_idx in connections:
                            to_route_id = int(to_trip_str.split("_")[0])
                            if to_stop_idx >= len(stoptimes_dict.get(to_route_id, [[]])):
                                continue
                            _enqueue_trip(to_trip_str, to_stop_idx,
                                          R_t, Q, n + 1,
                                          stoptimes_dict, MAX_TRANSFER)

        # --- Update best results for this departure time ---
        for desti in DESTINATION_LIST:
            best_round = None
            best_arr   = inf_time
            for r in range(1, MAX_TRANSFER + 1):
                if J[desti][r][0] < best_arr:
                    best_arr   = J[desti][r][0]
                    best_round = r

            if best_round is None:
                continue

            travel_td = best_arr - dep_time
            if travel_td.total_seconds() <= 0:
                continue

            if desti not in best or travel_td < best[desti][0]:
                best[desti] = (
                    travel_td,
                    best_round - 1,          # transfers = rounds - 1
                    J[desti][best_round][1]  # last trip id string
                )

    # --- Format output ---
    results = {}
    for desti, (travel_td, n_transfers, last_trip) in best.items():
        results[desti] = {
            "travel_time_minutes": round(travel_td.total_seconds() / 60, 2),
            "transfers":           n_transfers,
            "last_trip_id":        last_trip,
        }

    return results