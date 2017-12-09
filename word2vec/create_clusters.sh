#!/bin/sh
DATA_DIR=$1
D=128

MODEL="${DATA_DIR}/word2vec.d${D}"
STOPW_OUT="${DATA_DIR}/stopw_word2vec.txt"
CL_OUT="${DATA_DIR}/word_clusters.out"
CL_PKL="${DATA_DIR}/word_clusters.pkl"
W2CL_PKL="${DATA_DIR}/word2cluster.pkl"

python create_clusters.py -word2vec $MODEL -stopw $STOPW_OUT -cluster_out $CL_OUT -cluster_pkl $CL_PKL -word2cluster_pkl $W2CL_PKL
