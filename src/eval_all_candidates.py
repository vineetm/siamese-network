'''
Goal is to generate correlation scores for txt1 and [gt, candidates]
Thus, if there are 5000 candidates, we generates [datum x 5001] scores.
'''

import argparse, os, time
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import namedtuple
from utils import load_hparams
from tensorflow.python.ops import lookup_ops

logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model_dir')
  parser.add_argument('-best_model_dir', default='best_eval/')
  parser.add_argument('-candidates', help='All unique training candidates')
  parser.add_argument('-txt1', help='Input sentence, for example conversation context')
  parser.add_argument('-gt', help='Ground Truth candidate')
  parser.add_argument('-candidate_batch_size', default=512, type=int,
                      help='Batch size to compute RNN vectors for Candidates')
  args = parser.parse_args()
  return args


class SentenceDatum(namedtuple('SentenceDatum', 'initializer txt1 len_txt1')):
  pass


def create_single_file_iterator(vocab_table, file_name, batch_size):
  dataset = tf.data.TextLineDataset(file_name)
  dataset = dataset.map(lambda sentence: tf.string_split([sentence]).values)
  dataset = dataset.map(lambda words: vocab_table.lookup(words))
  dataset = dataset.map(lambda words: (words, tf.size(words)))

  dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
  iterator = dataset.make_initializable_iterator()

  txt1, len_txt1 = iterator.get_next()
  return SentenceDatum(iterator.initializer, txt1, len_txt1)


class EvalModel:
  def __init__(self, hparams, iterator, compute_scores=False):
    self.hparams = hparams
    self.iterator = iterator

    #To compute RNN vectors, we need W and rnn_cell and dynamic_rnn
    self.W = tf.get_variable('embeddings', shape=[self.hparams.size_vocab, self.hparams.d])

    self.txt1_vectors = tf.nn.embedding_lookup(self.W, self.iterator.txt1)

    rnn_cell = rnn.BasicLSTMCell(self.hparams.d, self.hparams.forget_bias)
    with tf.variable_scope('rnn'):
      _, state_txt1 = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt1_vectors, sequence_length=self.iterator.len_txt1,
                                        dtype=tf.float32)
    self.vec_txt1 = state_txt1.h

    self.saver = tf.train.Saver(tf.global_variables())

  def compute_txt1_vectors(self, sess):
    sess.run(self.iterator.initializer)
    start_time = time.time()
    all_vectors = []
    num_batches = 0
    while True:
      try:
        all_vectors.extend(sess.run(self.vec_txt1))
        num_batches += 1
        if num_batches % 100 == 0:
          logging.info('Num_batches: %d Sentences: %d'%(num_batches, len(all_vectors)))
      except tf.errors.OutOfRangeError:
        return all_vectors, time.time() - start_time


def get_candidate_vectors(vocab_table, args, hparams):
  # Create candidates iterator
  candidates_iterator = create_single_file_iterator(vocab_table, args.candidates, args.candidate_batch_size)
  model = EvalModel(hparams, candidates_iterator)
  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(os.path.join(args.model_dir, args.best_model_dir))
    model.saver.restore(sess, latest_ckpt)
    candidate_vectors, time_taken = model.compute_txt1_vectors(sess)
  logging.info('Computed %d Candidate vectors: Time: %ds'%(len(candidate_vectors), time_taken))
  del model
  return candidate_vectors


def main():
  args = setup_args()
  logging.info(args)

  #Let us first load hparams for the trained model
  hparams = load_hparams(os.path.join(args.model_dir, 'hparams'))
  logging.info(hparams)

  #Let us create vocab table next
  vocab_table = lookup_ops.index_table_from_file(hparams.vocab, default_value=0)
  candidate_vectors = get_candidate_vectors(vocab_table, args, hparams)




if __name__ == '__main__':
  main()