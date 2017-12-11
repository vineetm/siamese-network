#!/usr/bin/env bash
DATA_DIR=$1
BIN_REPR="${DATA_DIR}/candidates.bin.repr"
BIN="${DATA_DIR}/bin.txt"
CANDIDATES="${DATA_DIR}/candidates.txt"
CANDIDATES_BIN="${DATA_DIR}/candidates.txt.bin"

TRAIN_TXT1="${DATA_DIR}/train.txt1"
TRAIN_TXT2="${DATA_DIR}/train.txt2"
TRAIN_LABELS="${DATA_DIR}/train.labels"

TRAIN_TXT1_BIN="${DATA_DIR}/train.txt1.bin"
TRAIN_TXT2_BIN="${DATA_DIR}/train.txt2.bin"
TRAIN_LABELS_BIN="${DATA_DIR}/train.labels.bin"

VALID_TXT1="${DATA_DIR}/pvalid.txt1"


VALID_TXT1_BIN="${DATA_DIR}/svalid.txt1.bin"
VALID_TXT2_BIN="${DATA_DIR}/svalid.txt2.bin"
VALID_LABELS_BIN="${DATA_DIR}/svalid.labels.bin"


python create_bin_data.py -bin $BIN -binr $BIN_REPR -candidates $CANDIDATES -candidates_bin $CANDIDATES_BIN -txt1 $TRAIN_TXT1 -txt2 $TRAIN_TXT2 -labels $TRAIN_LABELS -out_txt1 $TRAIN_TXT1_BIN -out_txt2 $TRAIN_TXT2_BIN -out_labels $TRAIN_LABELS_BIN

python create_bin_data.py -bin $BIN -binr $BIN_REPR -candidates $CANDIDATES -candidates_bin $CANDIDATES_BIN -txt1 $VALID_TXT1 -txt2 $VALID_TXT2 -out_txt1 $VALID_TXT1_BIN -out_txt2 $VALID_TXT2_BIN -out_labels $VALID_LABELS_BIN