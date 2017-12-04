#!/bin/sh
DATA_DIR=$1
OUT_DIR=$2

mkdir -p $OUT_DIR
TRAIN_CSV="${DATA_DIR}/train.csv"
TRAIN_TXT1="${OUT_DIR}/train.txt1"
TRAIN_TXT2="${OUT_DIR}/train.txt2"
TRAIN_LABELS="${OUT_DIR}/train.labels"

python prep_data.py -train -input_csv $TRAIN_CSV -out_txt1 $TRAIN_TXT1 -out_txt2 $TRAIN_TXT2 -out_labels $TRAIN_LABELS 
