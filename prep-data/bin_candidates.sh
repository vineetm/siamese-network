#!/bin/sh
DATA_DIR=$1

WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"
CANDIDATES="${DATA_DIR}/candidates.txt"
CANDIDATES_BIN="${DATA_DIR}/candidates.txt.bin"
BIN_MEMBERS="${DATA_DIR}/candidates.bin.members"

python bin_candidates.py -word_clusters $WORD_CLUSTERS -candidates $CANDIDATES -candidates_bin $CANDIDATES_BIN -bin_members $BIN_MEMBERS
