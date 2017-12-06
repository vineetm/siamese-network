#!/bin/sh
OUT_DIR=$1
K=$2
CANDIDATES_PKL="${OUT_DIR}/candidates.txt.pkl"

PVALID_TXT1="${OUT_DIR}/pvalid.txt1"
PVALID_TXT2="${OUT_DIR}/pvalid.txt2"

ALL_VALID_TXT1="${OUT_DIR}/all.valid.txt1"
ALL_VALID_TXT2="${OUT_DIR}/all.valid.txt2"
ALL_VALID_MAP="${OUT_DIR}/all.valid.map"
python create_retrieval_data.py -candidates_pkl $CANDIDATES_PKL -input_txt1 $PVALID_TXT1 -input_txt2 $PVALID_TXT2 -output_txt1 $ALL_VALID_TXT1 -output_txt2 $ALL_VALID_TXT2 -output_map $ALL_VALID_MAP -k $K

PTEST_TXT1="${OUT_DIR}/ptest.txt1"
PTEST_TXT2="${OUT_DIR}/ptest.txt2"

ALL_TEST_TXT1="${OUT_DIR}/all.test.txt1"
ALL_TEST_TXT2="${OUT_DIR}/all.test.txt2"
ALL_TEST_MAP="${OUT_DIR}/all.test.map"

python create_retrieval_data.py -candidates_pkl $CANDIDATES_PKL -input_txt1 $PTEST_TXT1 -input_txt2 $PTEST_TXT2 -output_txt1 $ALL_TEST_TXT1 -output_txt2 $ALL_TEST_TXT2 -output_map $ALL_TEST_MAP -k $K


