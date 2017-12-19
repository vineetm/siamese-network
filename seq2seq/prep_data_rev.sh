#!/usr/bin/env bash
DATA_DIR=$1
STOPW="${DATA_DIR}/stopw_word2vec.txt"
WORD_CLUSTERS="${DATA_DIR}/word_clusters.out"

TRAIN_TXT1="${DATA_DIR}/train.txt1"
TRAIN_TXT2="${DATA_DIR}/train.txt2"
TRAIN_LABELS="${DATA_DIR}/train.labels"

OUT_TRAIN_TXT1="${DATA_DIR}/train.rev.seq2seq.txt1"
OUT_TRAIN_TXT2="${DATA_DIR}/train.rev.seq2seq.txt2"
OUT_TRAIN_INDEX="${DATA_DIR}/train.rev.seq2seq.index"

VALID_TXT1="${DATA_DIR}/pvalid.txt1"
VALID_TXT2="${DATA_DIR}/pvalid.txt2"

OUT_VALID_TXT1="${DATA_DIR}/valid.rev.seq2seq.txt1"
OUT_VALID_TXT2="${DATA_DIR}/valid.rev.seq2seq.txt2"
OUT_VALID_INDEX="${DATA_DIR}/valid.rev.seq2seq.index"

VOCAB_TXT1="${DATA_DIR}/vocab.rev.seq2seq.txt1"
VOCAB_TXT2="${DATA_DIR}/vocab.rev.seq2seq.txt2"

ALL_VOCAB_TXT1="${DATA_DIR}/all.vocab.rev.seq2seq.txt1"
ALL_VOCAB_TXT2="${DATA_DIR}/all.vocab.rev.seq2seq.txt2"


python prep_data.py -stopw $STOPW -word_clusters $WORD_CLUSTERS -txt1 $TRAIN_TXT1 -txt2 $TRAIN_TXT2 -labels $TRAIN_LABELS -out_txt1 $OUT_TRAIN_TXT1 -out_txt2 $OUT_TRAIN_TXT2 -out_index $OUT_TRAIN_INDEX -rev
echo "Tr data done"

python create_vocab.py -input $OUT_TRAIN_TXT1 -all_vocab $ALL_VOCAB_TXT1 -vocab $VOCAB_TXT1
python create_vocab.py -input $OUT_TRAIN_TXT2 -all_vocab $ALL_VOCAB_TXT2 -vocab $VOCAB_TXT2 -min_count 1
echo "Vocab done"

python prep_data.py -stopw $STOPW -word_clusters $WORD_CLUSTERS -txt1 $VALID_TXT1 -txt2 $VALID_TXT2 -out_txt1 $OUT_VALID_TXT1 -out_txt2 $OUT_VALID_TXT2 -out_index $OUT_VALID_INDEX -rev
