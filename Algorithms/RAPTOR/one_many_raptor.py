# Algorithms/RAPTOR/one_many_raptor.py
import pandas as pd
import numpy as np
from collections import defaultdict


def onetomany_raptor(
    SOURCE: int,
    DESTINATION_LIST: list,
    filter_time: pd.Timestamp,
    d_time_groups,
    MAX_TRANSFER: int,
    WALKING_FROM_SOURCE: int,
    PRINT_ITINERARY: int,
    OPTIMIZED: int,
    routes_by_stop_dict: dict,
    stops_dict: dict,
    stoptimes_dict: dict,
    footpath_dict: dict,
    idx_by_route_stop_dict: dict,
    trip_transfer_dict: dict,
    trip_set: set,
    MAX_TOTAL_TIME_MIN: float,
    MAX_WALK_TIME_MIN: float,
    ENABLE_JOURNEY_LOGGING: int = 0,
    log_file_path: str = None,
) -> dict:
    """
    One-to-many RAPTOR implementation.

    Args:
        SOURCE (int): stop id of source stop.
        DESTINATION_LIST (list): list of stop ids of destination stops.
        filter_time (pd.Timestamp): query departure time.
        d_time_groups: unused, kept for API symmetry with TBTR.
        MAX_TRANSFER (int): maximum number of transfers (rounds).
        WALKING_FROM_SOURCE (int): 1 = allow initial walk from SOURCE via footpaths.
        PRINT_ITINERARY (int): unused, kept for API compatibility.
        OPTIMIZED (int): unused, kept for API compatibility.
        routes_by_stop_dict (dict): {stop_id: [route_id, ...]}.
        stops_dict (dict): {route_id: [stop_id, ...]} stop sequence per route.
        stoptimes_dict (dict): {route_id: [trip_0, trip_1, ...]} trips sorted by
                               departure time; each trip is [(stop_id, arrival_time), ...].
        footpath_dict (dict): {from_stop_id: [(to_stop_id, timedelta), ...]}.
        idx_by_route_stop_dict (dict): {(route_id, stop_id): stop_index_in_route}.
        trip_transfer_dict (dict): unused, kept for API compatibility.
        trip_set (set): unused, kept for API compatibility.
        MAX_TOTAL_TIME_MIN (float): maximum total journey time in minutes.
        MAX_WALK_TIME_MIN (float): maximum cumulative walking time in minutes.
        ENABLE_JOURNEY_LOGGING (int): unused, kept for API compatibility.
        log_file_path (str | None): unused, kept for API compatibility.

    Returns:
        dict: Same format as onetomany_rtbtr (TBTR) for drop-in compatibility:
            {
                dest_id (int): {
                    "travel_time_minutes": float,
                    "transfers": int,
                    "last_trip_id": None,  # RAPTOR does not track trip ids
                }, ...
            }
    """
    MAX_TOTAL_TIME = pd.Timedelta(minutes=MAX_TOTAL_TIME_MIN)
    MAX_WALK_TIME  = pd.Timedelta(minutes=MAX_WALK_TIME_MIN)

    # Use a fixed far-future sentinel that is always valid regardless of run date.
    # BUG FIX: previously used pd.Timestamp('2099-12-31') which is fine, but the
    # TBTR side used a hardcoded 2026 date that is now in the past. Both now use
    # the same style of static far-future constant.
    INF = pd.Timestamp("2099-12-31 23:59:59")

    # ------------------------------------------------------------------ #
    # RAPTOR data structures                                               #
    # tau[k][stop]          = earliest arrival at stop using exactly k trips
    # earliest_arrival[stop] = best arrival across all rounds (for pruning)
    # labels[k][stop]        = dict with total_walk_time (used in post-proc)
    # ------------------------------------------------------------------ #
    tau              = {k: defaultdict(lambda: INF) for k in range(MAX_TRANSFER + 2)}
    labels           = {k: {}                       for k in range(MAX_TRANSFER + 2)}
    earliest_arrival = defaultdict(lambda: INF)
    marked_stops     = {SOURCE}

    # Round 0 — initialise source
    tau[0][SOURCE]              = filter_time
    earliest_arrival[SOURCE]    = filter_time
    labels[0][SOURCE]           = {"total_walk_time": pd.Timedelta(0)}

    if WALKING_FROM_SOURCE == 1:
        for target, walk_t in footpath_dict.get(SOURCE, []):
            arr = filter_time + walk_t
            if arr < earliest_arrival[target]:
                earliest_arrival[target] = arr
                tau[0][target]           = arr
                marked_stops.add(target)
                labels[0][target]        = {"total_walk_time": walk_t}

    # ------------------------------------------------------------------ #
    # RAPTOR rounds                                                        #
    # ------------------------------------------------------------------ #
    for k in range(1, MAX_TRANSFER + 1):
        if not marked_stops:
            break

        # Step A — collect the earliest marked stop on each route
        marked_routes: dict = {}
        for p in marked_stops:
            for route in routes_by_stop_dict.get(p, []):
                p_idx = idx_by_route_stop_dict[(route, p)]
                if route not in marked_routes or p_idx < marked_routes[route]["idx"]:
                    marked_routes[route] = {"stop": p, "idx": p_idx}

        marked_stops_next: set = set()

        # Step B — traverse each marked route
        for route, board_info in marked_routes.items():
            start_idx   = board_info["idx"]
            route_stops = stops_dict[route]
            route_trips = stoptimes_dict[route]

            current_trip_idx  = None
            board_stop_idx    = None
            prev_round_stop   = None
            current_prev_walk = pd.Timedelta(0)

            for i in range(start_idx, len(route_stops)):
                pi = route_stops[i]

                # --- try to alight at pi and improve its label ---
                if current_trip_idx is not None:
                    arr_time = route_trips[current_trip_idx][i][1]
                    if arr_time < earliest_arrival[pi]:
                        if arr_time - filter_time <= MAX_TOTAL_TIME:
                            if current_prev_walk <= MAX_WALK_TIME:
                                earliest_arrival[pi] = arr_time
                                tau[k][pi]           = arr_time
                                labels[k][pi]        = {
                                    "total_walk_time": current_prev_walk,
                                }
                                marked_stops_next.add(pi)

                # --- try to board an earlier/better trip at pi ---
                if tau[k - 1][pi] < INF:
                    # BUG FIX: find the earliest trip (smallest departure time at pi)
                    # that departs >= tau[k-1][pi].  Because stoptimes_dict is sorted
                    # by ascending departure time, the first matching index is correct.
                    # Previously the code compared trip *indices* (earliest_t < current_trip_idx)
                    # to decide whether to update, which is only safe because of the sort order.
                    # We keep that logic but make the intent explicit with a comment.
                    earliest_t = None
                    for t_idx, trip in enumerate(route_trips):
                        if trip[i][1] >= tau[k - 1][pi]:
                            earliest_t = t_idx
                            break   # trips are sorted; first match is earliest

                    if earliest_t is not None:
                        # Update only if this trip departs earlier than the one we're on
                        # (smaller index = earlier departure because trips are time-sorted)
                        if current_trip_idx is None or earliest_t < current_trip_idx:
                            current_trip_idx  = earliest_t
                            board_stop_idx    = i
                            prev_round_stop   = pi
                            current_prev_walk = (
                                labels[k - 1]
                                .get(pi, {})
                                .get("total_walk_time", pd.Timedelta(0))
                            )

        # Step C — traverse footpaths from newly reached stops
        for p in list(marked_stops_next):
            arr_p = tau[k][p]
            prev_walk_p = labels[k].get(p, {}).get("total_walk_time", pd.Timedelta(0))

            for target, walk_t in footpath_dict.get(p, []):
                arr_target = arr_p + walk_t
                if arr_target >= earliest_arrival[target]:
                    continue
                if arr_target - filter_time > MAX_TOTAL_TIME:
                    continue
                new_walk = prev_walk_p + walk_t
                if new_walk > MAX_WALK_TIME:
                    continue

                earliest_arrival[target] = arr_target
                tau[k][target]           = arr_target
                labels[k][target]        = {"total_walk_time": new_walk}
                marked_stops_next.add(target)

        marked_stops = marked_stops_next

    # ------------------------------------------------------------------ #
    # Post-process: pick best round per destination                        #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Post-process: pick best round per destination                        #
    # Output format matches onetomany_rtbtr (TBTR):                       #
    #   {dest_id (int): {travel_time_minutes, transfers, last_trip_id}}    #
    # ------------------------------------------------------------------ #
    results = {}

    for desti in DESTINATION_LIST:
        best_time           = None
        best_transfer_count = None
        best_round          = None

        for n in range(1, MAX_TRANSFER + 1):
            if tau[n][desti] >= INF:
                continue
            label = labels[n].get(desti)
            if not isinstance(label, dict):
                continue

            arrival_time = tau[n][desti]
            transfers    = n - 1

            if (
                best_time is None
                or arrival_time < best_time
                or (arrival_time == best_time and transfers < best_transfer_count)
            ):
                best_time           = arrival_time
                best_transfer_count = transfers
                best_round          = n

        if best_round is None:
            continue

        total_walk_seconds = (
            labels[best_round]
            .get(desti, {})
            .get("total_walk_time", pd.Timedelta(0))
            .total_seconds()
        )

        if total_walk_seconds > MAX_WALK_TIME.total_seconds():
            continue

        results[desti] = {
            "travel_time_minutes": round(
                (best_time - filter_time).total_seconds() / 60, 2),
            "transfers":           best_transfer_count,
            "last_trip_id":        None,   # RAPTOR does not track trip ids
        }

    return results