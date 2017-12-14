#!/usr/bin/env bash
DATA_DIR=$1
STOP_WORDS="${DATA_DIR}/stopw_word2vec.txt"
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"
CANDIDATES="${DATA_DIR}/candidates.txt"
CANDIDATES_BIN="${DATA_DIR}/candidates.txt.bin"


python bin_candidates.py -stopw $STOP_WORDS -word_clusters $WORD_CLUSTERS -candidates $CANDIDATES -candidates_bin $CANDIDATES_BIN -max_partial_len 4 -min_count 10