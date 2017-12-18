#!/usr/bin/env bash
DATA_DIR=$1
STOPW="${DATA_DIR}/stopw_word2vec.txt"

python prep_data.py -stopw $STOPW