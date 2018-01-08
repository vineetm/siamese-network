#!/usr/bin/env bash
DATA_DIR="${HOME}/data/ubuntu/data"
SEQ2SEQ_DATA_DIR="${HOME}/data/ubuntu/seq2seq-data"

TRAIN_TXT1="${DATA_DIR}/train.txt1"
TRAIN_TXT2="${DATA_DIR}/train.txt2"

WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"

OUT_TRAIN_TXT1="${SEQ2SEQ_DATA_DIR}/train.concepts.txt1"
OUT_TRAIN_TXT2="${SEQ2SEQ_DATA_DIR}/train.concepts.txt2"

python create_data_with_concepts.py -word_clusters $WORD_CLUSTERS -input_txt1 $TRAIN_TXT1 -input_txt2 $TRAIN_TXT2 -output_txt1 $OUT_TRAIN_TXT1 -output_txt2 $OUT_TRAIN_TXT2