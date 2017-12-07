'''
Goal is to generate correlation scores for txt1 and [gt, candidates]
Thus, if there are 5000 candidates, we generates [datum x 5001] scores.
'''

import argparse, os
import tensorflow as tf
from utils import load_hparams

logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model_dir')
  parser.add_argument('-candidates', help='All unique training candidates')
  parser.add_argument('-txt1', help='Input sentence, for example conversation context')
  parser.add_argument('-gt', help='Ground Truth candidate')
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  #Let us first load hparams for the trained model
  hparams = load_hparams(os.path.join(args.model_dir, 'hparams'))
  logging.info(hparams)



if __name__ == '__main__':
  main()