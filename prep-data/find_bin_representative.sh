#!/bin/sh
DATA_DIR=$1
BIN_MEMBERS="${DATA_DIR}/candidates.bin.members"
CANDIDATES="${DATA_DIR}/candidates.txt"
CANDIDATES_BIN="${DATA_DIR}/candidates.txt.bin"
VOCAB_FILE="${DATA_DIR}/vocab.10k.txt"
BIN_REPR="${DATA_DIR}/candidates.bin.repr"
BIN="${DATA_DIR}/bin.txt"

python find_bin_representative.py -candidates $CANDIDATES -candidates_bin $CANDIDATES_BIN -bin_members $BIN_MEMBERS -vocab_file $VOCAB_FILE -bin_repr $BIN_REPR -bin $BIN