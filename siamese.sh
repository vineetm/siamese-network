#!/bin/sh
python siamese.py -data_dir data -vocab_suffix vocab100k.txt -train_batch_size 128 -steps_per_stats 100 -steps_per_eval 1000
