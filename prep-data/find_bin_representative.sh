#!/bin/sh
DATA_DIR=$1
BIN_MEMBERS="${DATA_DIR}/candidates.bin.members"
CANDIDATES="${DATA_DIR}/candidates.txt"
VOCAB_FILE="${DATA_DIR}/vocab.10k.txt"
BIN_REPR="${DATA_DIR}/candidates.bin.repr"

python find_bin_representative.py -candidates $CANDIDATES -bin_members $BIN_MEMBERS -vocab_file $VOCAB_FILE -bin_repr $BIN_REPR