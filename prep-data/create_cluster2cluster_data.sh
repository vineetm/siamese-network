#!/bin/sh
DATA_DIR=$1
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"

TRAIN_TXT1="${DATA_DIR}/train.txt1"
TRAIN_TXT2="${DATA_DIR}/train.txt2"
TRAIN_LABELS="${DATA_DIR}/train.labels"

OUT_TRAIN_TXT1="${DATA_DIR}/train.c2c.txt1"
OUT_TRAIN_TXT2="${DATA_DIR}/train.c2c.txt2"

VOCAB_TXT1="${DATA_DIR}/vocab.c2c.txt1"
VOCAB_TXT2="${DATA_DIR}/vocab.c2c.txt2"

python create_cluster2cluster_data.py -word_clusters $WORD_CLUSTERS -txt1 $TRAIN_TXT1 -txt2 $TRAIN_TXT2 -labels $TRAIN_LABELS -out_txt1 $OUT_TRAIN_TXT1 -out_txt2 $OUT_TRAIN_TXT2 -vocab $VOCAB_TXT1
cp $VOCAB_TXT1 $VOCAB_TXT2