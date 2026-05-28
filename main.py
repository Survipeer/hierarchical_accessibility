"""
Main orchestrator for the Hierarchical Accessibility Calculation Workflow.
"""
import os
import sys

python_global_call = 'python'

def run_step(script_path, description):
    print(f"\n{'='*40}")
    print(f"STEP: {description}")
    print(f"{'='*40}")
    exit_code = os.system(f'{python_global_call} {script_path}')
    if exit_code != 0:
        print(f"Error encountered in {script_path}. Exiting.")
        sys.exit(1)

# Phase 1: Base Spatial & Network Setup
run_step('GTFS_wrapper.py', "GTFS Preprocessing")
run_step('./builders/build_transfer_file.py', "Building Transfers")
run_step('hex_hash_map.py', "Generating Hexagons")
run_step('walk_time.py', "Calculating Ingress Walk Times")

# Phase 2: Hierarchical Setup
run_step('hierarchical_setup.py', "Regionalization & Gate Identification")

# Phase 3: Region-Aware Pre-calculation
run_step('./builders/build_TBTR_dict.py', "Building Region-Aware TBTR Dictionaries")

run_step('regional_precalc.py', "Pre-calculating Regional Transit Chunks")

# Phase 4: Analysis
run_step('build_walk_only_hex2hex_osm.py', "Building OSM Walk-Only Baseline")
run_step('reachability_hierarchical.py', "Running Hierarchical Reachability Analysis")

print("\nAll Steps Completed Successfully.")
