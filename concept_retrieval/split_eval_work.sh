#!/usr/bin/env bash
WORD_CLUSTERS="${HOME}/data/ubuntu/data/word_clusters.out"
CANDIDATES="${HOME}/data/ubuntu/data/candidates.txt"

CL_MAP="${HOME}/data/ubuntu/data/word_clusters.map"
CL_COUNTS="${HOME}/data/ubuntu/data/word_clusters.counts"

TXT1="${HOME}/data/ubuntu/data/pvalid.txt1"

PREDS_CANDIDATES_PKL=$1
SCORES_DIR=$2

mkdir -p $SCORES_DIR
python split_eval_work.py -candidates $CANDIDATES -txt1 $TXT1 -preds_candidates_pkl $PREDS_CANDIDATES_PKL -sdir $SCORES_DIR