#!/bin/sh
DATA_DIR=$1
OUT_DIR=$2

mkdir -p $OUT_DIR

#Prepare training data for siamese
TRAIN_CSV="${DATA_DIR}/train.csv"
TRAIN_TXT1="${OUT_DIR}/train.txt1"
TRAIN_TXT2="${OUT_DIR}/train.txt2"
TRAIN_LABELS="${OUT_DIR}/train.labels"
python prep_data.py -train -input_csv $TRAIN_CSV -out_txt1 $TRAIN_TXT1 -out_txt2 $TRAIN_TXT2 -out_labels $TRAIN_LABELS 

#Prepare validation data for siamese
VALID_CSV="${DATA_DIR}/valid.csv"
SIAMESE_VALID_TXT1="${OUT_DIR}/svalid.txt1"
SIAMESE_VALID_TXT2="${OUT_DIR}/svalid.txt2"
SIAMESE_VALID_LABELS="${OUT_DIR}/svalid.labels"
python prep_data.py -siamese_valid -input_csv $VALID_CSV -out_txt1 $SIAMESE_VALID_TXT1 -out_txt2 $SIAMESE_VALID_TXT2 -out_labels $SIAMESE_VALID_LABELS 
