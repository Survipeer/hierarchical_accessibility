HIERARCHICAL TRANSIT ACCESSIBILITY WORKFLOW

OVERVIEW
--------------------------------------------------------------------------------
This repository contains a robust Python-based geospatial workflow for 
calculating hierarchical transit accessibility using GTFS (General Transit Feed 
Specification) and OpenStreetMap (OSM) data. 

The project optimizes large-scale public transit routing queries by dividing 
networks into regional chunks, establishing "gates" between regions, and 
pre-calculating time-dependent transit paths. It utilizes highly efficient 
routing algorithms including TBTR (Trip-Based Transit Routing), RAPTOR, and 
Time-Expanded Dijkstra.


KEY FEATURES
--------------------------------------------------------------------------------
* Hierarchical Regionalization: Uses spatial partitioning and clustering to 
  divide massive transit networks into smaller, manageable regions.
* Fast Multimodal Routing: Calculates combined walk + transit travel times.
* Hexagonal Spatial Indexing: Uses Uber's H3 resolution grids for uniform 
  spatial origin-destination matrices.
* Time-Dependent Gate Graph: Evaluates dynamic transfer budgets and travel 
  times across inter-region gates using a custom lookup stitcher and modified 
  Dijkstra approach.


PIPELINE ARCHITECTURE
--------------------------------------------------------------------------------
The workflow is orchestrated via 'main.py' and operates in four distinct phases:

1. Phase 1: Base Spatial & Network Setup - Preprocesses GTFS datasets.
   - Generates H3 hexagonal grids.
   - Computes pedestrian ingress/egress walk times using OSM networks.
2. Phase 2: Hierarchical Setup
   - Clusters stops into regions and identifies boundary "gates".
3. Phase 3: Region-Aware Pre-calculation
   - Builds regional TBTR dictionaries.
   - Pre-calculates intra-region, ingress, egress, and connecting transit chunks.
4. Phase 4: Analysis
   - Builds an OSM walk-only baseline.
   - Executes the final hierarchical reachability analysis across the network.


PROJECT STRUCTURE
--------------------------------------------------------------------------------
Algorithms/               # Routing algorithms (RAPTOR, TBTR, TE_DIJKSTRA)
Data/                     # (Ignored in git) Place your GTFS and OSM data here
Tools/                    # Helper tools like visualizers and DBScan tuners
builders/                 # Scripts to build dictionaries and transfers
main.py                   # Main orchestrator script
requirements.txt          # Python dependencies
.gitignore                # Git ignore rules


INSTALLATION
--------------------------------------------------------------------------------
1. Clone the repository to your local machine.
2. Ensure you have Python 3.10+ installed.
3. Install the required packages by running the following command in your terminal:
   pip install -r requirements.txt


DATA PREPARATION
--------------------------------------------------------------------------------
1. Create a directory for your transit data: ./Data/GTFS/{NETWORK_NAME}
2. Place your raw GTFS feed (stops.txt, trips.txt, stop_times.txt, routes.txt, 
   etc.) inside this directory. You can also place the raw .zip file in the 
   main directory to be processed by GTFS_wrapper.py.
3. Set your specific network parameters if prompted during script execution.


USAGE
--------------------------------------------------------------------------------
To execute the complete accessibility calculation pipeline, run:
python main.py

Note: Depending on the size of your GTFS network, building the initial transfer 
dictionaries and pre-calculating regional chunks may take significant time and 
memory. Progress is logged in the terminal.


ALGORITHMS & ROUTING
--------------------------------------------------------------------------------
This project incorporates several state-of-the-art transit routing algorithms:
* TBTR (Trip-Based Transit Routing): Core engine for evaluating time-dependent 
  transit paths.
* RAPTOR (Round-Based Public Transit Routing): Alternative engine for specific 
  routing analyses.
* Hierarchical Stitcher: A pure lookup query system that stitches parallel 
  intra-region and inter-region (gate-to-gate) paths while strictly enforcing 
  global transfer budgets.


LICENSE
--------------------------------------------------------------------------------
This project is licensed under the GNU General Public License v3.0. 
See the LICENSE file for more details.
