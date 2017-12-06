#!/bin/sh
OUT_DIR=$1

#These are the candidates
TRAIN_TXT2="${OUT_DIR}/train.txt2"

#We need labels as we only pick (flag=1) as a candidate
TRAIN_LABELS="${OUT_DIR}/train.labels"

OUT_CANDIDATES="${OUT_DIR}/candidates.txt"
OUT_CANDIDATES_PKL="${OUT_DIR}/candidates.txt.pkl"

python store_train_candidates.py -all_candidates $TRAIN_TXT2 -all_labels $TRAIN_LABELS -out_candidates_txt $OUT_CANDIDATES -out_candidates_pkl $OUT_CANDIDATES_PKL
