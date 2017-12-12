#!/bin/sh
DATA_DIR=$1
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"

TRAIN_TXT1="${DATA_DIR}/train.txt1"
TRAIN_TXT2="${DATA_DIR}/train.txt2"
TRAIN_LABELS="${DATA_DIR}/train.labels"

OUT_TRAIN_TXT1="${DATA_DIR}/train.uniq.c2c.txt1"
OUT_TRAIN_TXT2="${DATA_DIR}/train.uniq.c2c.txt2"

VOCAB_TXT1="${DATA_DIR}/vocab.uniq.c2c.txt1"
VOCAB_TXT2="${DATA_DIR}/vocab.uniq.c2c.txt2"

PVALID_TXT1="${DATA_DIR}/pvalid.txt1"
PVALID_TXT2="${DATA_DIR}/pvalid.txt2"

OUT_VALID_TXT1="${DATA_DIR}/valid.uniq.c2c.txt1"
OUT_VALID_TXT2="${DATA_DIR}/valid.uniq.c2c.txt2"

PTEST_TXT1="${DATA_DIR}/ptest.txt1"
PTEST_TXT2="${DATA_DIR}/ptest.txt2"

OUT_TEST_TXT1="${DATA_DIR}/test.uniq.c2c.txt1"
OUT_TEST_TXT2="${DATA_DIR}/test.uniq.c2c.txt2"


echo "Creating train data"
python create_cluster2cluster_data.py -word_clusters $WORD_CLUSTERS -txt1 $TRAIN_TXT1 -txt2 $TRAIN_TXT2 -labels $TRAIN_LABELS -out_txt1 $OUT_TRAIN_TXT1 -out_txt2 $OUT_TRAIN_TXT2 -out_vocab $VOCAB_TXT1 -uniq
cp $VOCAB_TXT1 $VOCAB_TXT2

echo "Creating valid data"
python create_cluster2cluster_data.py -word_clusters $WORD_CLUSTERS -txt1 $PVALID_TXT1 -txt2 $PVALID_TXT2 -out_txt1 $OUT_VALID_TXT1 -out_txt2 $OUT_VALID_TXT2 -uniq

echo "Creating test data"
python create_cluster2cluster_data.py -word_clusters $WORD_CLUSTERS -txt1 $PTEST_TXT1 -txt2 $PTEST_TXT2 -out_txt1 $OUT_TEST_TXT1 -out_txt2 $OUT_TEST_TXT2 -uniq

echo "Done!"