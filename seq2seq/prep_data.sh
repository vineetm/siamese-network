#!/usr/bin/env bash
DATA_DIR=$1
STOPW="${DATA_DIR}/stopw_word2vec.txt"
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"

python prep_data.py -stopw $STOPW -word_clusters $WORD_CLUSTERS