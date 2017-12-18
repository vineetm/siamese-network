#!/usr/bin/env bash
DATA_DIR=$1
STOPW="${DATA_DIR}/stopw_word2vec.txt"
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"

TRAIN_TXT1="${DATA_DIR}/train.txt1"
TRAIN_TXT2="${DATA_DIR}/train.txt2"
TRAIN_LABELS="${DATA_DIR}/train.labels"

OUT_TRAIN_TXT1="${DATA_DIR}/train.seq2seq.txt1"
OUT_TRAIN_TXT2="${DATA_DIR}/train.seq2seq.txt2"
OUT_TRAIN_INDEX="${DATA_DIR}/train.seq2seq.index"


python prep_data.py -stopw $STOPW -word_clusters $WORD_CLUSTERS -txt1 $TRAIN_TXT1 -txt2 $TRAIN_TXT2 -labels $TRAIN_LABELS -out_txt1 $OUT_TRAIN_TXT1 -out_txt2 $OUT_TRAIN_TXT2 -out_index $OUT_TRAIN_INDEX