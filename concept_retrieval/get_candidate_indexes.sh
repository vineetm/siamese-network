#!/usr/bin/env bash
WORD_CLUSTERS="${HOME}/data/ubuntu/data/word_clusters.out"
CANDIDATES="${HOME}/data/ubuntu/data/candidates.txt"

CL_MAP="${HOME}/data/ubuntu/data/word_clusters.map"

PREDS=$1
PREDS_CANDIDATES_PKL=$2

python get_candidate_indexes.py -word_clusters $WORD_CLUSTERS -cluster_map $CL_MAP -preds $PREDS -preds_candidates_pkl $PREDS_CANDIDATES_PKL