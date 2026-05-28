"""
hierarchical_query.py  —  HierarchicalStitcher

Pure lookup stitcher. No TBTR calls at query time.
All four precomputed dicts are loaded once in __init__.

Journey model
-------------
Two parallel paths are computed and merged at query time:

PATH A — Direct intra-region (non-gate origins only)
    Starting Stop (non-gate)
        ↓ intra_chunk[stop][bucket][dest_stop]   direct within same region
    Destination Stop

PATH B — Gate-based (all origins)
    Starting Stop
        ↓ ingress_chunk[stop][bucket][gate]          intra-region
    Origin Gate
        ↓ gate_pairs[gate][bucket][next_gate]        inter-region (crosses boundary)
    Entry Gate of Transit Region
        ↓ connecting_chunk[gate][bucket][exit_gate]  intra-region (through transit region)
    Exit Gate of Transit Region
        ↓ gate_pairs[gate][bucket][next_gate]        inter-region
        ... repeat as needed ...
    Destination Gate
        ↓ egress_chunk[gate][bucket][stop]           intra-region
    Final Stop

Results from both paths are merged, keeping the best travel time per destination stop.

Global transfer budget
----------------------
MAX_TRANSFER is a single budget shared across ALL phases:
    ingress_transfers + gate_crossings + connecting_transfers + egress_transfers
    <= MAX_TRANSFER

This prevents unrealistic stacking of transfers across phases.
The chunks are precomputed with a generous MAX_TRANSFER so they cover
all possible journeys; the budget is enforced here at query time only.

Time-dependent Dijkstra
------------------------
At each gate, two edge types are expanded:
  1. gate_pairs     — board a trip crossing into another region (+1 transfer)
  2. connecting_chunks — travel to another gate within the same region
Both are skipped if they would exceed the global transfer budget.
"""

import os
import sys
import pickle
import heapq

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Time bucket helper
# ---------------------------------------------------------------------------

_BUCKET_LABELS  = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0,15,30,45)]
_BUCKET_SECONDS = [h*3600 + m*60 for h in range(24) for m in (0,15,30,45)]


def nearest_bucket(dt) -> str:
    """Return nearest 15-min bucket label for a datetime/Timestamp."""
    ts  = pd.Timestamp(dt)
    sec = ts.hour * 3600 + ts.minute * 60 + ts.second
    best_i    = 0
    best_diff = abs(sec - _BUCKET_SECONDS[0])
    for i, bs in enumerate(_BUCKET_SECONDS[1:], 1):
        diff = abs(sec - bs)
        if diff < best_diff:
            best_diff = diff
            best_i    = i
    return _BUCKET_LABELS[best_i]


def add_minutes(dt, minutes: float) -> pd.Timestamp:
    return pd.Timestamp(dt) + pd.Timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# Stitcher
# ---------------------------------------------------------------------------

class HierarchicalStitcher:

    def __init__(self, NETWORK_NAME):
        self.network_name = NETWORK_NAME
        h_path = f"./Data/Hierarchical/{NETWORK_NAME}"

        print(f"[Stitcher] Loading chunks...")

        with open(f"{h_path}/ingress_chunks.pkl", "rb") as f:
            self.ingress_chunks = pickle.load(f)

        with open(f"{h_path}/egress_chunks.pkl", "rb") as f:
            self.egress_chunks = pickle.load(f)

        with open(f"{h_path}/connecting_chunks.pkl", "rb") as f:
            self.connecting_chunks = pickle.load(f)

        with open(f"{h_path}/gate_pairs.pkl", "rb") as f:
            self.gate_pairs = pickle.load(f)

        with open(f"{h_path}/intra_chunks.pkl", "rb") as f:
            self.intra_chunks = pickle.load(f)

        print(f"[Stitcher] Ready.")

    # ------------------------------------------------------------------
    # Time-dependent Dijkstra over gate graph
    # ------------------------------------------------------------------

    def _gate_dijkstra(self, origin_gate_arrivals, departure_time,
                       max_total_time, max_transfer):
        """
        Time-dependent Dijkstra over the gate graph.

        Expands two edge types per gate:
          - gate_pairs:        inter-region trip segment (+1 transfer)
          - connecting_chunks: intra-region gate-to-gate

        Both are skipped if they would exceed max_transfer.

        Args:
            origin_gate_arrivals: {gate_id: {total_minutes, arrival_time,
                                             ingress_walk, transfers}}
            departure_time: pd.Timestamp
            max_total_time: float (minutes)
            max_transfer: int — global budget remaining after ingress

        Returns:
            {gate_id: {total_minutes, arrival_time, transfers, ingress_walk}}
        """
        # pq entries: (total_minutes, transfers, gate_id, arrival_time, ingress_walk)
        pq   = []
        best = {}

        for gate_id, info in origin_gate_arrivals.items():
            heapq.heappush(pq, (
                info["total_minutes"],
                info["transfers"],
                gate_id,
                info["arrival_time"],
                info["ingress_walk"],
            ))
            best[gate_id] = info.copy()

        while pq:
            total_min, transfers, current_gate, current_arr, ingress_walk = \
                heapq.heappop(pq)

            # Stale entry
            if total_min > best.get(current_gate, {}).get(
                    "total_minutes", float("inf")):
                continue

            if total_min >= max_total_time:
                continue

            # No point expanding if transfer budget already exhausted
            if transfers >= max_transfer:
                continue

            bucket = nearest_bucket(current_arr)

            # --- Edge type 1: gate_pairs (inter-region) ---
            for dst_gate, pair_data in (self.gate_pairs
                                        .get(current_gate, {})
                                        .get(bucket, {})
                                        .items()):
                dst_gate      = int(dst_gate)
                # Boarding a new inter-region trip = 1 transfer
                new_transfers = transfers + 1
                if new_transfers > max_transfer:
                    continue

                new_total = total_min + pair_data["travel_time_minutes"]
                if new_total >= max_total_time:
                    continue

                if (dst_gate not in best or
                        new_total < best[dst_gate]["total_minutes"]):
                    arr = add_minutes(current_arr,
                                      pair_data["travel_time_minutes"])
                    best[dst_gate] = {
                        "total_minutes": new_total,
                        "arrival_time":  arr,
                        "transfers":     new_transfers,
                        "ingress_walk":  ingress_walk,
                    }
                    heapq.heappush(pq, (new_total, new_transfers,
                                        dst_gate, arr, ingress_walk))

            # --- Edge type 2: connecting_chunks (intra-region) ---
            for exit_gate, chunk_data in (self.connecting_chunks
                                          .get(current_gate, {})
                                          .get(bucket, {})
                                          .items()):
                exit_gate     = int(exit_gate)
                new_transfers = transfers + chunk_data["transfers"]
                if new_transfers > max_transfer:
                    continue

                new_total = total_min + chunk_data["travel_time_minutes"]
                if new_total >= max_total_time:
                    continue

                if (exit_gate not in best or
                        new_total < best[exit_gate]["total_minutes"]):
                    arr = add_minutes(current_arr,
                                      chunk_data["travel_time_minutes"])
                    best[exit_gate] = {
                        "total_minutes": new_total,
                        "arrival_time":  arr,
                        "transfers":     new_transfers,
                        "ingress_walk":  ingress_walk,
                    }
                    heapq.heappush(pq, (new_total, new_transfers,
                                        exit_gate, arr, ingress_walk))

        return best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stitched_results(self,
                             origin_stops,
                             departure_time,
                             max_walking=10,
                             max_total_time=60,
                             max_transfer=3):
        """
        Find all reachable destination stops from a set of walkable stops.

        Args:
            origin_stops (list[dict]): each has 'stop_id' and
                'walking_time_minutes'.
            departure_time: when traveller leaves the hex centroid.
            max_walking (float): max total walking allowed (minutes).
            max_total_time (float): max total journey time (minutes).
            max_transfer (int): global transfer budget across ALL phases.

        Returns:
            {dest_stop_id(int): {
                'total_time': float,
                'transfers': int,
                'ingress_walk': float,
            }}
        """
        departure_time = pd.Timestamp(departure_time)

        # ----------------------------------------------------------------
        # Phase 0 — INTRA-REGION DIRECT: non-gate stop -> dest stop
        # Handles journeys entirely within one region that never touch a gate.
        # Only non-gate origin stops are considered here; gate stops are
        # covered by the ingress->egress path below.
        # ----------------------------------------------------------------
        intra_results = {}

        for stop_info in origin_stops:
            stop_id      = int(stop_info["stop_id"])
            ingress_walk = float(stop_info["walking_time_minutes"])

            if ingress_walk > max_walking:
                continue
            if stop_id not in self.intra_chunks:
                continue  # gate stop or stop with no intra data

            arr_at_stop = add_minutes(departure_time, ingress_walk)
            bucket      = nearest_bucket(arr_at_stop)

            for dest_stop, chunk in (self.intra_chunks
                                     .get(stop_id, {})
                                     .get(bucket, {})
                                     .items()):
                dest_stop  = int(dest_stop)
                total_time = ingress_walk + chunk["travel_time_minutes"]
                transfers  = chunk["transfers"]

                if total_time >= max_total_time:
                    continue
                if transfers > max_transfer:
                    continue

                if (dest_stop not in intra_results or
                        total_time < intra_results[dest_stop]["total_time"]):
                    intra_results[dest_stop] = {
                        "total_time":   round(total_time, 2),
                        "transfers":    transfers,
                        "ingress_walk": ingress_walk,
                    }

        # ----------------------------------------------------------------
        # Phase 1 — INGRESS: starting stop -> origin gate
        # ----------------------------------------------------------------
        origin_gate_arrivals = {}

        for stop_info in origin_stops:
            stop_id      = int(stop_info["stop_id"])
            ingress_walk = float(stop_info["walking_time_minutes"])

            if ingress_walk > max_walking:
                continue

            arr_at_stop = add_minutes(departure_time, ingress_walk)
            bucket      = nearest_bucket(arr_at_stop)

            for gate_id, chunk in (self.ingress_chunks
                                   .get(stop_id, {})
                                   .get(bucket, {})
                                   .items()):
                gate_id           = int(gate_id)
                ingress_transfers = chunk["transfers"]

                # Ingress already consumes part of the budget
                if ingress_transfers > max_transfer:
                    continue

                total_min = ingress_walk + chunk["travel_time_minutes"]
                if total_min >= max_total_time:
                    continue

                if (gate_id not in origin_gate_arrivals or
                        total_min < origin_gate_arrivals[gate_id]["total_minutes"]):
                    origin_gate_arrivals[gate_id] = {
                        "total_minutes": total_min,
                        "arrival_time":  add_minutes(departure_time, total_min),
                        "ingress_walk":  ingress_walk,
                        "transfers":     ingress_transfers,
                    }

        if not origin_gate_arrivals:
            return {}

        # ----------------------------------------------------------------
        # Phase 2 — DIJKSTRA over gate graph
        # Remaining budget per gate = max_transfer - ingress_transfers
        # The Dijkstra tracks cumulative transfers and enforces the budget
        # ----------------------------------------------------------------
        dest_gate_best = self._gate_dijkstra(
            origin_gate_arrivals = origin_gate_arrivals,
            departure_time       = departure_time,
            max_total_time       = max_total_time,
            max_transfer         = max_transfer,
        )

        # ----------------------------------------------------------------
        # Phase 3 — EGRESS: destination gate -> final stop
        # Enforce remaining budget: gate_transfers + egress_transfers <= max_transfer
        # ----------------------------------------------------------------
        final_results = {}

        for dest_gate, gate_data in dest_gate_best.items():
            dest_gate        = int(dest_gate)
            gate_transfers   = gate_data["transfers"]
            remaining        = max_transfer - gate_transfers
            bucket           = nearest_bucket(gate_data["arrival_time"])

            for dest_stop, egress_chunk in (self.egress_chunks
                                            .get(dest_gate, {})
                                            .get(bucket, {})
                                            .items()):
                dest_stop        = int(dest_stop)
                egress_transfers = egress_chunk["transfers"]

                # Enforce global budget
                if egress_transfers > remaining:
                    continue

                total_time      = (gate_data["total_minutes"] +
                                   egress_chunk["travel_time_minutes"])
                total_transfers = gate_transfers + egress_transfers

                if total_time >= max_total_time:
                    continue

                if (dest_stop not in final_results or
                        total_time < final_results[dest_stop]["total_time"]):
                    final_results[dest_stop] = {
                        "total_time":   round(total_time, 2),
                        "transfers":    total_transfers,
                        "ingress_walk": gate_data["ingress_walk"],
                    }

        # ----------------------------------------------------------------
        # Merge Phase 0 (intra) into gate-based results
        # Keep best travel time per destination stop across both paths
        # ----------------------------------------------------------------
        for dest_stop, intra_data in intra_results.items():
            if (dest_stop not in final_results or
                    intra_data["total_time"] < final_results[dest_stop]["total_time"]):
                final_results[dest_stop] = intra_data

        return final_results