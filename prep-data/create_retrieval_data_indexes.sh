#!/bin/sh
OUT_DIR=$1
K=$2
CANDIDATES_PKL="${OUT_DIR}/candidates.txt.pkl"

PVALID_TXT2="${OUT_DIR}/pvalid.txt2"
ALL_VALID_MAP="${OUT_DIR}/all.valid.index_map"
python create_retrieval_data_indexes.py -candidates_pkl $CANDIDATES_PKL -gt $PVALID_TXT2 -output_map $ALL_VALID_MAP -k $K

PTEST_TXT2="${OUT_DIR}/ptest.txt2"
ALL_TEST_MAP="${OUT_DIR}/all.test.index_map"
python create_retrieval_data_indexes.py -candidates_pkl $CANDIDATES_PKL -gt $PTEST_TXT2 -output_map $ALL_TEST_MAP -k $K


