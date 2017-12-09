#!/bin/sh
DATA_DIR=$1
D=128

MODEL="${DATA_DIR}/word2vec.d${D}"
STOPW_OUT="${DATA_DIR}/stopw_word2vec.txt"
CL_OUT="${DATA_DIR}/word_clusters.out"


python create_clusters.py -word2vec $MODEL -stopw $STOPW_OUT -cluster_out $CL_OUT
