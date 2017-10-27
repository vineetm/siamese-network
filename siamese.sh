#!/bin/sh
python siamese.py -data_dir data -vocab_suffix vocab.txt -train_batch_size 32 -valid_batch_size 32 -steps_per_stats 10 -steps_per_eval 20 -vocab 30000
