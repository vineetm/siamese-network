#!/usr/bin/env bash
DATA_DIR=$1
MODEL_DIR=$2

BEAM_SIZE=$3

INFER_INP_FILE="${DATA_DIR}/valid.c2c.txt1"
INFER_GT="${DATA_DIR}/valid.c2c.txt2"
INFER_OUT_FILE="${MODEL_DIR}/valid.c2c.preds"

python -m nmt.nmt --inference_input_file $INFER_INP_FILE --inference_output_file $INFER_OUT_FILE --out_dir=${MODEL_DIR}/best_accuracy

python check_coverage.py -gt $INFER_GT -preds $INFER_OUT_FILE