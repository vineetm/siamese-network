#!/usr/bin/env bash

DATA_DIR=$1

TXT1="${DATA_DIR}/train.txt1"
TXT2="${DATA_DIR}/train.txt2"
LABELS="${DATA_DIR}/train.labels"

OUT_TXT1="${DATA_DIR}/train.seq2seq.txt1"
OUT_TXT2="${DATA_DIR}/train.seq2seq.txt2"

ALL_VOCAB_TXT1="${DATA_DIR}/all.vocab.seq2seq.txt1"
ALL_VOCAB_TXT2="${DATA_DIR}/all.vocab.seq2seq.txt2"

VOCAB_TXT1="${DATA_DIR}/vocab.seq2seq.txt1"
VOCAB_TXT2="${DATA_DIR}/vocab.seq2seq.txt2"

python create_seq2seq_data.py -txt1 $TXT1 -txt2 $TXT2 -labels $LABELS -out_txt1 $OUT_TXT1 -out_txt2 $OUT_TXT2

python create_vocab.py -txt1 $TXT1 -vocab $VOCAB_TXT1 -all_vocab $ALL_VOCAB_TXT1
python create_vocab.py -txt1 $TXT2 -vocab $VOCAB_TXT2 -all_vocab $ALL_VOCAB_TXT2    