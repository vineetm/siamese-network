import argparse
import tensorflow as tf
import numpy as np

logging = tf.logging
logging.set_verbosity(logging.INFO)

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('scores_file')
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  fr = open(args.scores_file)

  scores = []

  correct_1 = 0
  correct_2 = 0
  correct_5 = 0

  total = 0.0
  for line in fr:
    scores.append(float(line.strip()))

    if len(scores) == 10:
      total += 1
      ranks = np.argsort(scores)[::-1]

      correct_1 += np.sum(ranks[:1] == 0)
      correct_2 += np.sum(ranks[:2] == 0)
      correct_5 += np.sum(ranks[:5] == 0)
      scores = []

  logging.info('C@1:%d C@2: %d C@5:%d Total: %d' % (correct_1, correct_2, correct_5, total))
  logging.info('R@1:%.4f R@2: %.4f R@5:%.4f'%(correct_1/total, correct_2/total, correct_5/total))



if __name__ == '__main__':
  main()