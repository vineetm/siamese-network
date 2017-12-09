#!/bin/sh
DATA_DIR=$1
T=$2

D=128
FILES="${DATA_DIR}/train.txt1,${DATA_DIR}/train.txt2,${DATA_DIR}/train.labels"
MODEL="${DATA_DIR}/word2vec.d${D}"

python train_word2vec.py -files $FILES -model $MODEL -workers $T -d $D
