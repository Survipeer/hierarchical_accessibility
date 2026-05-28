"""
hierarchical_setup.py — Regionalize the transit network and identify gate stops.

Regionalization algorithms
--------------------------
1. Louvain       — Community detection on the transit graph (connectivity-based).
                   Non-geographic; stops sharing many routes end up together.
                   Non-deterministic unless random_state is fixed.
                   Does NOT accept an exact k; number of regions is emergent.

2. K-Means       — Clusters stop lat/lon coordinates into exactly k regions.
                   Fast, always produces k regions, but cluster shapes can be
                   irregular and ignores transit topology entirely.

3. Agglomerative — Hierarchical clustering on lat/lon, merged bottom-up by
                   geographic proximity. Produces compact, spatially contiguous
                   regions. Accepts exact k.

4. DBSCAN        — Density-based clustering on lat/lon. Automatically determines
                   the number of regions based on stop density; no k needed.
                   Stops in sparse areas may be labelled as noise (region = -1)
                   and are assigned to the nearest cluster.

Gate identification (same for all algorithms)
---------------------------------------------
A stop is a gate if it has at least one trip edge connecting it to a stop
in a different region. Gates are stored in BOTH the source region dict and
the gates pkl so that intra-region routing near boundaries still works.

Output files (./Data/Hierarchical/{NETWORK_NAME}/)
---------------------------------------------------
regions.pkl  — {stop_id (int): region_id (int)}
regions.json — same, string keys for inspection
gates.pkl    — {region_id (int): [gate_stop_id, ...]}
gates.json   — same, string keys for inspection
"""

import os
import sys
import json
import pickle
import warnings
from collections import defaultdict
from time import time as _now

import numpy as np
import pandas as pd
import networkx as nx


# ---------------------------------------------------------------------------
# Algorithm implementations
# ---------------------------------------------------------------------------

def _partition_louvain(G, stops_df, n_regions, random_state):
    """
    Louvain community detection on the transit graph.
    n_regions is used as a target via binary search on resolution if provided,
    otherwise Louvain picks the number of communities automatically.
    random_state fixes the seed for reproducibility.
    """
    try:
        from community import community_louvain
    except ImportError:
        raise ImportError(
            "python-louvain is required for Louvain partitioning.\n"
            "Install with: pip install python-louvain"
        )

    if n_regions is None:
        print("  Running Louvain (auto k)...")
        partition = community_louvain.best_partition(
            G, random_state=random_state)
        return partition

    # Binary search on resolution to hit the target n_regions
    print(f"  Running Louvain with binary search for k={n_regions}...")
    lo, hi   = 0.01, 20.0
    best_partition = None
    best_diff      = float("inf")

    for iteration in range(40):
        mid = (lo + hi) / 2
        partition = community_louvain.best_partition(
            G, resolution=mid, random_state=random_state)
        n = len(set(partition.values()))
        diff = abs(n - n_regions)

        if diff < best_diff:
            best_diff      = diff
            best_partition = partition

        if n == n_regions:
            print(f"    Exact match at resolution={mid:.4f} → {n} regions")
            return partition
        elif n < n_regions:
            lo = mid
        else:
            hi = mid

        if hi - lo < 1e-6:
            break

    actual = len(set(best_partition.values()))
    print(f"  Closest found: {actual} regions (target was {n_regions})")
    return best_partition


def _partition_kmeans(G, stops_df, n_regions, random_state):
    """
    K-Means clustering on stop lat/lon coordinates.
    Requires n_regions. Geographic — spatially compact clusters.
    """
    if n_regions is None:
        raise ValueError("K-Means requires n_regions to be specified.")

    from sklearn.cluster import KMeans

    stop_ids = list(G.nodes())
    coords   = _get_coords(stop_ids, stops_df)

    print(f"  Running K-Means (k={n_regions})...")
    km = KMeans(n_clusters=n_regions, random_state=random_state, n_init=10)
    labels = km.fit_predict(coords)

    return {stop_id: int(label)
            for stop_id, label in zip(stop_ids, labels)}


def _partition_agglomerative(G, stops_df, n_regions, random_state):
    """
    Agglomerative (hierarchical) clustering on stop lat/lon coordinates.
    Produces compact, spatially contiguous regions.
    Requires n_regions.
    """
    if n_regions is None:
        raise ValueError("Agglomerative clustering requires n_regions to be specified.")

    from sklearn.cluster import AgglomerativeClustering

    stop_ids = list(G.nodes())
    coords   = _get_coords(stop_ids, stops_df)

    print(f"  Running Agglomerative clustering (k={n_regions})...")
    agg = AgglomerativeClustering(n_clusters=n_regions, linkage="ward")
    labels = agg.fit_predict(coords)

    return {stop_id: int(label)
            for stop_id, label in zip(stop_ids, labels)}


def _partition_dbscan(G, stops_df, n_regions, random_state,
                      eps_km=1.0, min_samples=3):
    """
    DBSCAN clustering on stop lat/lon.
    n_regions is ignored — DBSCAN determines k automatically from density.
    eps_km: neighbourhood radius in kilometres.
    min_samples: minimum stops to form a core point.

    Noise stops (label = -1) are assigned to the nearest cluster centroid.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import pairwise_distances_argmin_min

    stop_ids = list(G.nodes())
    coords   = _get_coords(stop_ids, stops_df)

    # DBSCAN on radians so we can use haversine metric
    coords_rad = np.radians(coords)
    earth_r_km = 6371.0
    eps_rad    = eps_km / earth_r_km

    print(f"  Running DBSCAN (eps={eps_km}km, min_samples={min_samples})...")
    db     = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  DBSCAN found {n_clusters} clusters, {n_noise} noise stops")

    # Assign noise stops to nearest cluster centroid
    if n_noise > 0 and n_clusters > 0:
        cluster_mask     = labels != -1
        cluster_coords   = coords[cluster_mask]
        cluster_labels   = labels[cluster_mask]
        noise_mask       = labels == -1
        noise_coords     = coords[noise_mask]

        # Compute centroids
        unique_clusters  = np.unique(cluster_labels)
        centroids        = np.array([
            cluster_coords[cluster_labels == c].mean(axis=0)
            for c in unique_clusters
        ])

        # Nearest centroid for each noise stop
        noise_indices    = np.where(noise_mask)[0]
        for i, ni in enumerate(noise_indices):
            dists        = np.linalg.norm(centroids - noise_coords[i], axis=1)
            nearest_c    = unique_clusters[np.argmin(dists)]
            labels[ni]   = nearest_c

    if n_regions is not None:
        print(f"  Note: n_regions={n_regions} is ignored by DBSCAN. "
              f"Found {n_clusters} clusters instead.")

    return {stop_id: int(label)
            for stop_id, label in zip(stop_ids, labels)}


# ---------------------------------------------------------------------------
# Helper — extract lat/lon array aligned with a list of stop_ids
# ---------------------------------------------------------------------------

def _get_coords(stop_ids, stops_df):
    """
    Return a float32 array of shape (N, 2) — [lat, lon] — aligned with stop_ids.
    Stops missing from stops_df are placed at (0, 0) with a warning.
    """
    coord_map = dict(zip(
        stops_df["stop_id"].astype(int),
        zip(stops_df["stop_lat"].astype(float),
            stops_df["stop_lon"].astype(float))
    ))
    missing = [s for s in stop_ids if s not in coord_map]
    if missing:
        warnings.warn(f"{len(missing)} stops have no coordinates — placed at (0,0)")

    coords = np.array(
        [coord_map.get(s, (0.0, 0.0)) for s in stop_ids],
        dtype=np.float32
    )
    return coords


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "1": ("Louvain",       _partition_louvain),
    "2": ("K-Means",       _partition_kmeans),
    "3": ("Agglomerative", _partition_agglomerative),
    "4": ("DBSCAN",        _partition_dbscan),
}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def identify_regions_and_gates(NETWORK_NAME, algorithm="1",
                                n_regions=None, random_state=42,
                                dbscan_eps_km=1.0, dbscan_min_samples=3):
    """
    Partition the transit network into regions and identify gate stops.

    Parameters
    ----------
    NETWORK_NAME : str
    algorithm    : str — "1" Louvain | "2" K-Means | "3" Agglomerative | "4" DBSCAN
    n_regions    : int | None — target number of regions (ignored by DBSCAN)
    random_state : int — RNG seed for reproducibility
    dbscan_eps_km      : float — DBSCAN neighbourhood radius in km
    dbscan_min_samples : int   — DBSCAN minimum stops per core point
    """
    algo_name, algo_fn = ALGORITHMS[str(algorithm)]
    print(f"\nAlgorithm  : {algo_name}")
    print(f"n_regions  : {n_regions if n_regions else 'auto'}")
    print(f"random_state: {random_state}")

    path = f"./Data/GTFS/{NETWORK_NAME}"
    stops      = pd.read_csv(f"{path}/stops.txt")
    stop_times = pd.read_csv(f"{path}/stop_times.txt")

    # Ensure stop_id is int
    stops["stop_id"]           = stops["stop_id"].astype(int)
    stop_times["stop_id"]      = stop_times["stop_id"].astype(int)

    # ------------------------------------------------------------------
    # Build transit graph
    # Nodes = stops, edges = consecutive stop pairs on the same trip
    # ------------------------------------------------------------------
    print("Building transit graph...")
    G = nx.Graph()
    G.add_nodes_from(stops["stop_id"].tolist())

    for trip_id, group in stop_times.groupby("trip_id"):
        sorted_stops = (group.sort_values("stop_sequence")["stop_id"].tolist())
        for i in range(len(sorted_stops) - 1):
            G.add_edge(sorted_stops[i], sorted_stops[i + 1])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Run chosen algorithm
    # ------------------------------------------------------------------
    print("Partitioning network...")

    _t0 = _now()
    if algorithm == "4":  # DBSCAN
        partition = algo_fn(G, stops, n_regions, random_state,
                            eps_km=dbscan_eps_km,
                            min_samples=dbscan_min_samples)
    else:
        partition = algo_fn(G, stops, n_regions, random_state)
    partition_time_sec = _now() - _t0

    # partition: {stop_id (int): region_id (int)}
    actual_n = len(set(partition.values()))
    print(f"Regions produced : {actual_n}")
    print(f"Partition time   : {partition_time_sec:.2f}s")

    # ------------------------------------------------------------------
    # Identify gate stops
    # A stop is a gate if any trip edge connects it to a different region
    # ------------------------------------------------------------------
    gates_by_region = defaultdict(set)
    for u, v in G.edges():
        ru = partition.get(u, -1)
        rv = partition.get(v, -1)
        if ru != rv:
            if ru != -1:
                gates_by_region[ru].add(u)
            if rv != -1:
                gates_by_region[rv].add(v)

    total_gates = sum(len(gs) for gs in gates_by_region.values())
    print(f"Gate stops  : {total_gates} across {len(gates_by_region)} regions")

    # ------------------------------------------------------------------
    # Print region summary
    # ------------------------------------------------------------------
    stops_per_region = defaultdict(int)
    for stop_id, region_id in partition.items():
        stops_per_region[region_id] += 1

    print(f"\n{'Region':>8}  {'Stops':>6}  {'Gates':>6}")
    print("-" * 26)
    for rid in sorted(stops_per_region):
        n_stops = stops_per_region[rid]
        n_gates = len(gates_by_region.get(rid, []))
        print(f"{rid:>8}  {n_stops:>6}  {n_gates:>6}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = f"./Data/Hierarchical/{NETWORK_NAME}"
    os.makedirs(out_dir, exist_ok=True)

    gates_dict = {k: list(v) for k, v in gates_by_region.items()}

    with open(f"{out_dir}/regions.pkl", "wb") as f:
        pickle.dump(partition, f)
    with open(f"{out_dir}/regions.json", "w") as f:
        json.dump({str(k): v for k, v in partition.items()}, f)

    with open(f"{out_dir}/gates.pkl", "wb") as f:
        pickle.dump(gates_dict, f)
    with open(f"{out_dir}/gates.json", "w") as f:
        json.dump({str(k): v for k, v in gates_dict.items()}, f)

    print(f"\nSaved to {out_dir}/")
    print(f"  regions.pkl / regions.json")
    print(f"  gates.pkl   / gates.json")
    # Save timing record
    timing_record = {
        "network":            NETWORK_NAME,
        "algorithm":          algo_name,
        "n_regions_target":   n_regions,
        "n_regions_actual":   actual_n,
        "n_gate_stops":       total_gates,
        "n_graph_nodes":      G.number_of_nodes(),
        "n_graph_edges":      G.number_of_edges(),
        "random_state":       random_state,
        "partition_time_sec": round(partition_time_sec, 4),
    }
    timing_path = f"{out_dir}/setup_timing.json"

    # Append to existing log if present, otherwise start fresh
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            timing_log = json.load(f)
        if not isinstance(timing_log, list):
            timing_log = [timing_log]
    else:
        timing_log = []

    timing_log.append(timing_record)
    with open(timing_path, "w") as f:
        json.dump(timing_log, f, indent=2)

    print(f"  setup_timing.json (appended)")
    print(f"\nHierarchical setup complete.")
    print(f"  {actual_n} regions  |  {total_gates} gate stops")
    print(f"  Partition time: {partition_time_sec:.2f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("./parameters_entered.txt", "rb") as f:
        parameter_files = pickle.load(f)
    NETWORK_NAME = parameter_files[1]

    print("=" * 50)
    print("STEP: Hierarchical Region Setup")
    print("=" * 50)
    print(f"Network: {NETWORK_NAME}")

    print("\nChoose regionalization algorithm:")
    print("  1 — Louvain       (connectivity-based, auto k)")
    print("  2 — K-Means       (geographic, exact k required)")
    print("  3 — Agglomerative (geographic, exact k required, compact)")
    print("  4 — DBSCAN        (density-based, auto k)")
    algo = input(": ").strip() or "1"

    if algo not in ALGORITHMS:
        print(f"Invalid choice '{algo}', defaulting to Louvain (1).")
        algo = "1"

    algo_name = ALGORITHMS[algo][0]
    print(f"\nSelected: {algo_name}")

    # n_regions
    n_regions = None
    if algo in ("2", "3"):
        n_regions = int(input(f"Number of regions (k): ").strip())
    elif algo == "1":
        k_input = input(
            "Target number of regions (leave blank for auto): ").strip()
        n_regions = int(k_input) if k_input else None

    # random_state (not applicable to DBSCAN)
    random_state = 42
    if algo != "4":
        rs_input = input("Random state (default: 42): ").strip()
        random_state = int(rs_input) if rs_input else 42

    # DBSCAN-specific params
    dbscan_eps_km      = 1.0
    dbscan_min_samples = 3
    if algo == "4":
        eps_input = input(
            "DBSCAN neighbourhood radius in km (default: 1.0): ").strip()
        dbscan_eps_km = float(eps_input) if eps_input else 1.0

        ms_input = input(
            "DBSCAN min stops per core point (default: 3): ").strip()
        dbscan_min_samples = int(ms_input) if ms_input else 3

    identify_regions_and_gates(
        NETWORK_NAME      = NETWORK_NAME,
        algorithm         = algo,
        n_regions         = n_regions,
        random_state      = random_state,
        dbscan_eps_km     = dbscan_eps_km,
        dbscan_min_samples= dbscan_min_samples,
    )