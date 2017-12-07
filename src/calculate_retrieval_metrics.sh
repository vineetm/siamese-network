#!/bin/sh
MODEL_DIR=$1

SCORES="${MODEL_DIR}/valid.scores.pkl"
METRICS="${MODEL_DIR}/rk.metrics.out"

python calculate_retrieval_metrics.py -scores $SCORES -out_metrics $METRICS
