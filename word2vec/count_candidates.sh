#!/bin/sh
DATA_DIR=$1
CANDIDATES="${DATA_DIR}/candidates.txt"
CL_OUT="${DATA_DIR}/word_clusters.out"
CANDIDATE_MAP="${DATA_DIR}/candidates.map"
CANDIDATES_MISSED="${DATA_DIR}/candidates.missed"

python count_candidates.py -candidates $CANDIDATES -cluster_out $CL_OUT -candidate_map $CANDIDATE_MAP -candidates_missed $CANDIDATES_MISSED
