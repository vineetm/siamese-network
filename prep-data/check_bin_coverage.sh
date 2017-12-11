#!/bin/sh
DATA_DIR=$1

BIN="${DATA_DIR}/bin.txt"
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"
SENTENCES="${DATA_DIR}/pvalid.txt2"

python check_bin_coverage.py -bin $BIN -word_clusters $WORD_CLUSTERS -sentences $SENTENCES