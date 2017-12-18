#!/usr/bin/env bash
DATA_DIR=$1

WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"
CANDIDATES="${DATA_DIR}/candidates.txt"
CANDIDATES_BIN="${DATA_DIR}/candidates.txt.bin"
BIN_CLUSTERS="${DATA_DIR}/bin.clusters.txt"
BIN_WORDS="${DATA_DIR}/bin.words.txt"
BIN_MEMBERS="${DATA_DIR}/bin.members.txt"
BIN_SAMPLES="${DATA_DIR}/bin.samples.txt"
BIN_COUNTS="${DATA_DIR}/bin.counts.txt"

python gather_bin_candidates.py -candidates_bin $CANDIDATES_BIN -word_clusters $WORD_CLUSTERS -candidates $CANDIDATES -bin_clusters $BIN_CLUSTERS -bin_words $BIN_WORDS -bin_counts $BIN_COUNTS -bin_members $BIN_MEMBERS -bin_samples $BIN_SAMPLES