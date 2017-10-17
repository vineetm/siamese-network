'''
Input: a) Vocab file b) Pre-trained word vectors (glove)
Output: Numpy array with initialized wordvectors
'''

import argparse
import tensorflow as tf
import numpy as np

np.random.seed(1543)
logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('vocab_file')
  parser.add_argument('word2vec_file')
  parser.add_argument('out')

  parser.add_argument('-d', default=300, type=int)

  args = parser.parse_args()
  return args


def get_word2index(vocab_file):
  word2index = {}
  index = 0
  with open(vocab_file) as fr:
    for line in fr:
      word = line.strip()
      word2index[word] = index
  return word2index


def get_word_vectors(word2index, word2vec_file, d):
  V = len(word2index)
  W_np = np.random.uniform(-0.01, 0.01, [V, d]).astype(np.float32)

  numFound = 0
  with open(word2vec_file) as fr:
    for line in  fr:
      parts = line.split()


      if len(parts) != (d+1):
        continue

      word = parts[0].strip()
      if word in word2index:
        W_np[word2index[word]] = [np.float32(v) for v in parts[1:]]
        numFound += 1

  logging.info('Found: %d V:%d'%(numFound, V))
  return W_np


def main():
  args = setup_args()
  logging.info(args)

  word2index = get_word2index(args.vocab_file)
  W_np = get_word_vectors(word2index, args.word2vec_file, args.d)

  np.save(args.out, W_np)


if __name__ == '__main__':
  main()