#!/bin/sh
MODEL_DIR='out'
DATA_DIR='../data'
mkdir -p $MODEL_DIR

python train.py $DATA_DIR $MODEL_DIR  
