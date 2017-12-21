#!/usr/bin/env bash
DATA_DIR=$1
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"
CANDIDATES="${DATA_DIR}/candidates.txt"
CLUSTER_CANDIDATES="${DATA_DIR}/cluster_candidates.txt"

python build_candidate_cluster_map.py -word_clusters $WORD_CLUSTERS -candidates $CANDIDATES -cluster_candidates $CLUSTER_CANDIDATES