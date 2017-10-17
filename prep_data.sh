#!/bin/sh

RAW_DATA_DIR='/Users/vineet/repos/github/ubuntu-ranking-dataset-creator/src'
OUT_DIR='data-tmp'

mkdir -p $OUT_DIR

python prep_data.py $RAW_DATA_DIR $OUT_DIR
