#!/bin/sh
DATA_DIR=$1

#These are the candidates
TRAIN_TXT2="${DATA_DIR}/train.txt2"

#We need labels as we only pick (flag=1) as a candidate
TRAIN_LABELS="${DATA_DIR}/train.labels"

#This is where we write all the candidates
OUT_CANDIDATES="${DATA_DIR}/candidates.txt"

python store_train_candidates.py -all_candidates $TRAIN_TXT2 -all_labels $TRAIN_LABELS -out_candidates $OUT_CANDIDATES