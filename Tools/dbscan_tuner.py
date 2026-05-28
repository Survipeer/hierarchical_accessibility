"""
dbscan_tuner.py — Find DBSCAN parameters to hit a target cluster count.

Usage
-----
    python dbscan_tuner.py

Reads stops.txt from ./Data/GTFS/{NETWORK_NAME}/stops.txt and runs a
systematic grid search over (eps_km, min_samples) pairs to find all
parameter combinations that produce the target number of clusters.

Output
------
  1. Console summary table of every (eps, min_samples) combination tried.
  2. A JSON file  ./Data/Hierarchical/{NETWORK_NAME}/dbscan_tuner_results.json
     with the full grid search results and the recommended parameter set.

Algorithm
---------
  Phase 1 — coarse sweep:
    eps_km in [0.1, 0.2, 0.3, ..., 5.0]  (step 0.1)
    min_samples in [2, 3, 4, 5, 7, 10]
    Builds a table of (eps_km, min_samples) → n_clusters.

  Phase 2 — fine sweep around promising eps ranges:
    For each min_samples value where the coarse sweep crossed the target,
    binary-search eps_km to 0.01 km precision.

  Phase 3 — report:
    List every (eps_km, min_samples) pair that hits the target ± tolerance.
    Recommend the pair with the lowest noise count among exact matches,
    or the lowest noise count within ±1 cluster if no exact match exists.

Notes
-----
  - Noise stops (label = -1) are NOT counted as a cluster.
  - Noise stops are reassigned to the nearest cluster centroid in the
    actual hierarchical_setup.py run, so they do not create orphan stops.
  - Cluster count can be non-monotone in eps for sparse networks; the
    binary search handles this by scanning densely in ambiguous regions.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stops(network_name: str) -> np.ndarray:
    """Load stop coordinates from GTFS stops.txt. Returns (N, 2) float array [lat, lon]."""
    path = f"./Data/GTFS/{network_name}/stops.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"stops.txt not found at {path}")
    stops = pd.read_csv(path)
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    coords = stops[["stop_lat", "stop_lon"]].values
    print(f"Loaded {len(coords)} stops from {path}")
    return coords


def run_dbscan(coords_rad: np.ndarray, eps_km: float, min_samples: int) -> tuple:
    """
    Run DBSCAN and return (n_clusters, n_noise, labels).
    coords_rad: (N, 2) array in radians.
    """
    eps_rad = eps_km / 6371.0
    labels  = DBSCAN(eps=eps_rad, min_samples=min_samples,
                     metric="haversine", algorithm="ball_tree").fit_predict(coords_rad)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    return n_clusters, n_noise, labels


def coarse_sweep(coords_rad, target, min_samples_list, eps_range, eps_step):
    """
    Sweep eps_km values for each min_samples and record (n_clusters, n_noise).
    Returns a dict: {min_samples: [(eps_km, n_clusters, n_noise), ...]}
    """
    results = {}
    eps_values = np.round(np.arange(eps_range[0], eps_range[1] + eps_step, eps_step), 3)
    total = len(min_samples_list) * len(eps_values)
    done  = 0

    for ms in min_samples_list:
        results[ms] = []
        for eps in eps_values:
            n_clust, n_noise, _ = run_dbscan(coords_rad, eps, ms)
            results[ms].append((float(eps), n_clust, n_noise))
            done += 1
            print(f"\r  Coarse sweep: {done}/{total} — eps={eps:.2f}km  "
                  f"min_samples={ms}  clusters={n_clust}  noise={n_noise}    ",
                  end="", flush=True)

    print()
    return results


def find_crossing_intervals(sweep_list, target, tolerance):
    """
    From a coarse sweep list [(eps, n_clusters, n_noise), ...],
    find eps intervals where n_clusters crosses within target ± tolerance.
    Returns list of (eps_lo, eps_hi) intervals.
    """
    intervals = []
    for i in range(len(sweep_list) - 1):
        eps_lo, n_lo, _ = sweep_list[i]
        eps_hi, n_hi, _ = sweep_list[i + 1]
        lo_ok = abs(n_lo - target) <= tolerance
        hi_ok = abs(n_hi - target) <= tolerance
        # interval is interesting if either end is close, or the target lies between
        crosses = (n_lo - target) * (n_hi - target) <= 0  # sign change
        if lo_ok or hi_ok or crosses:
            intervals.append((eps_lo, eps_hi))
    return intervals


def fine_binary_search(coords_rad, target, ms, eps_lo, eps_hi, n_steps=20):
    """
    Dense linear scan within [eps_lo, eps_hi] at fine resolution.
    Returns list of (eps_km, n_clusters, n_noise) for all steps.
    """
    eps_values = np.round(np.linspace(eps_lo, eps_hi, n_steps), 4)
    results = []
    for eps in eps_values:
        n_clust, n_noise, _ = run_dbscan(coords_rad, eps, ms)
        results.append((float(eps), n_clust, n_noise))
    return results


def recommend(candidates, target):
    """
    From a list of (eps, min_samples, n_clusters, n_noise) candidates,
    pick the best recommendation:
      1. Exact match with lowest noise
      2. If no exact match, closest cluster count with lowest noise
    """
    exact   = [(e, ms, nc, nn) for e, ms, nc, nn in candidates if nc == target]
    if exact:
        return min(exact, key=lambda x: x[3])  # lowest noise among exact

    if candidates:
        closest_nc = min(abs(nc - target) for _, _, nc, _ in candidates)
        close      = [(e, ms, nc, nn) for e, ms, nc, nn in candidates
                      if abs(nc - target) == closest_nc]
        return min(close, key=lambda x: x[3])

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DBSCAN Parameter Tuner")
    print("=" * 60)

    # Read parameters
    with open("./parameters_entered.txt", "rb") as f:
        parameter_files = pickle.load(f)
    NETWORK_NAME = parameter_files[1]
    print(f"Network: {NETWORK_NAME}")

    target     = int(input("\nTarget number of clusters (regions): ").strip())
    tolerance  = int(input("Acceptable tolerance (e.g. 1 means target ± 1): ").strip() or "1")

    print("\nMin-samples values to try (comma-separated, default: 2,3,4,5,7,10):")
    ms_input = input(": ").strip() or "2,3,4,5,7,10"
    min_samples_list = [int(x.strip()) for x in ms_input.split(",")]

    print("\nCoarse sweep eps range in km (default: 0.1 to 5.0):")
    eps_lo_input  = input("  Start eps_km (default 0.1): ").strip() or "0.1"
    eps_hi_input  = input("  End   eps_km (default 5.0): ").strip() or "5.0"
    eps_step_in   = input("  Step  eps_km (default 0.1): ").strip() or "0.1"
    eps_lo        = float(eps_lo_input)
    eps_hi        = float(eps_hi_input)
    eps_step      = float(eps_step_in)

    # Load stops
    coords = load_stops(NETWORK_NAME)
    coords_rad = np.radians(coords)

    print(f"\nTarget clusters  : {target} (tolerance ± {tolerance})")
    print(f"Min-samples list : {min_samples_list}")
    print(f"Coarse eps range : {eps_lo} — {eps_hi} km  step={eps_step}")
    print(f"Total coarse calls: {len(min_samples_list) * int((eps_hi - eps_lo) / eps_step + 1)}")
    print()

    # ── Phase 1: Coarse sweep ────────────────────────────────────────────────
    print("Phase 1: Coarse sweep...")
    coarse = coarse_sweep(coords_rad, target, min_samples_list,
                          (eps_lo, eps_hi), eps_step)

    # ── Phase 2: Fine search around crossing intervals ────────────────────────
    print("\nPhase 2: Fine search around promising intervals...")
    all_candidates = []   # (eps_km, min_samples, n_clusters, n_noise)

    # Collect coarse candidates first
    for ms, sweep in coarse.items():
        for eps, nc, nn in sweep:
            if abs(nc - target) <= tolerance:
                all_candidates.append((eps, ms, nc, nn))

    # Fine search
    for ms, sweep in coarse.items():
        intervals = find_crossing_intervals(sweep, target, tolerance + 1)
        for iv_lo, iv_hi in intervals:
            fine = fine_binary_search(coords_rad, target, ms, iv_lo, iv_hi, n_steps=25)
            for eps, nc, nn in fine:
                if abs(nc - target) <= tolerance:
                    all_candidates.append((eps, ms, nc, nn))
            print(f"  min_samples={ms}  eps in [{iv_lo:.2f}, {iv_hi:.2f}]  "
                  f"→ {len([x for x in fine if abs(x[1]-target) <= tolerance])} hits")

    # Deduplicate candidates (round eps to 2dp to avoid float noise)
    seen = set()
    unique_candidates = []
    for eps, ms, nc, nn in sorted(all_candidates, key=lambda x: (x[2] != target, x[3], x[0])):
        key = (round(eps, 2), ms)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((round(eps, 2), ms, nc, nn))

    # ── Phase 3: Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS — target={target}  tolerance=±{tolerance}")
    print("=" * 60)

    exact   = [(e, ms, nc, nn) for e, ms, nc, nn in unique_candidates if nc == target]
    close   = [(e, ms, nc, nn) for e, ms, nc, nn in unique_candidates if nc != target]

    if exact:
        print(f"\nExact matches ({len(exact)} found):")
        print(f"  {'eps_km':>8}  {'min_samples':>11}  {'clusters':>8}  {'noise':>6}")
        print("  " + "-" * 40)
        for eps, ms, nc, nn in sorted(exact, key=lambda x: x[3]):
            print(f"  {eps:>8.2f}  {ms:>11}  {nc:>8}  {nn:>6}")
    else:
        print(f"\nNo exact matches found for target={target}.")

    if close:
        print(f"\nClose matches (within tolerance ±{tolerance}):")
        print(f"  {'eps_km':>8}  {'min_samples':>11}  {'clusters':>8}  {'noise':>6}")
        print("  " + "-" * 40)
        for eps, ms, nc, nn in sorted(close, key=lambda x: (abs(x[2]-target), x[3])):
            print(f"  {eps:>8.2f}  {ms:>11}  {nc:>8}  {nn:>6}")

    rec = recommend(unique_candidates, target)
    if rec:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION:")
        print(f"  eps_km      = {rec[0]}")
        print(f"  min_samples = {rec[1]}")
        print(f"  clusters    = {rec[2]}")
        print(f"  noise stops = {rec[3]}")
        print(f"\nUse these values when running hierarchical_setup.py with DBSCAN.")
        print(f"{'='*60}")
    else:
        print("\nNo candidates found. Try widening the eps range or tolerance.")

    # ── Coarse sweep overview table ─────────────────────────────────────────
    print("\n\nCoarse sweep overview (cluster counts):")
    eps_vals = sorted(set(round(e, 2) for ms_data in coarse.values() for e, _, _ in ms_data))
    header   = f"{'eps_km':>8}  " + "  ".join(f"ms={ms:>2}" for ms in min_samples_list)
    print(header)
    print("-" * len(header))
    for eps in eps_vals:
        row = f"{eps:>8.2f}  "
        for ms in min_samples_list:
            match = next((nc for e, nc, _ in coarse[ms] if round(e, 2) == eps), None)
            marker = " *" if match is not None and abs(match - target) <= tolerance else "  "
            row += f"{match if match is not None else '--':>5}{marker}"
        print(row)
    print(f"\n  * = within {tolerance} of target ({target})")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_dir  = f"./Data/Hierarchical/{NETWORK_NAME}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/dbscan_tuner_results.json"

    output = {
        "network":      NETWORK_NAME,
        "target":       target,
        "tolerance":    tolerance,
        "n_stops":      int(len(coords)),
        "recommendation": {
            "eps_km":      rec[0] if rec else None,
            "min_samples": rec[1] if rec else None,
            "n_clusters":  rec[2] if rec else None,
            "n_noise":     rec[3] if rec else None,
        } if rec else None,
        "exact_matches": [
            {"eps_km": e, "min_samples": ms, "n_clusters": nc, "n_noise": nn}
            for e, ms, nc, nn in exact
        ],
        "close_matches": [
            {"eps_km": e, "min_samples": ms, "n_clusters": nc, "n_noise": nn}
            for e, ms, nc, nn in close
        ],
        "coarse_sweep": {
            str(ms): [{"eps_km": e, "n_clusters": nc, "n_noise": nn}
                      for e, nc, nn in data]
            for ms, data in coarse.items()
        }
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()