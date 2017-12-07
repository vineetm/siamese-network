#!/bin/sh

MODEL_DIR=$1
DATA_DIR=$2

CANDIDATES="${DATA_DIR}/candidates.txt"
TXT1="${DATA_DIR}/pvalid.txt1"
GT="${DATA_DIR}/pvalid.txt2"
MAP="${DATA_DIR}/all.valid.index_map"
SCORES="${MODEL_DIR}/valid.scores.pkl"
METRICS="${MODEL_DIR}/rk.metrics.out"

python eval_all_candidates.py -model_dir $MODEL_DIR -candidates $CANDIDATES -gt $GT -map $MAP -txt1 $TXT1 -scores_pkl $SCORES -out_metrics $METRICS
