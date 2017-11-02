import numpy as np
import tensorflow as tf
import argparse

logging = tf.logging
logging.set_verbosity(logging.INFO)
np.random.seed(1543)

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('start', type=int)
  parser.add_argument('end', type=int)

  parser.add_argument('num_samples', type=int)
  args = parser.parse_args()

  return args


def main():
  args = setup_args()
  logging.info(args)

  #Generates a number between 0 and 1
  r = np.random.rand(args.num_samples)

  #Scale this
  r = r * (args.end - args.start)

  #Add start
  r = args.start + r

  r = 10 ** r

  logging.info(r)

if __name__ == '__main__':
  main()
