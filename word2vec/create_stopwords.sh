#!/bin/sh
DATA_DIR=$1

CANDIDATES="${DATA_DIR}/candidates.txt"
STOPW_OUT="${DATA_DIR}/stopw_word2vec.txt"

python create_stopwords.py -stopw_out $STOPW_OUT -candidates $CANDIDATES
