#!/usr/bin/env bash
WORD_CLUSTERS="${HOME}/data/ubuntu/data/word_clusters.out"
CANDIDATES="${HOME}/data/ubuntu/data/candidates.txt"

CL_MAP="${HOME}/data/ubuntu/data/word_clusters.map"
CL_COUNTS="${HOME}/data/ubuntu/data/word_clusters.counts"

python build_candidate_map.py -word_clusters $WORD_CLUSTERS -candidates $CANDIDATES -cluster_map $CL_MAP -cluster_count $CL_COUNTS