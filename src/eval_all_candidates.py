'''
Goal is to generate correlation scores for txt1 and [gt, candidates]
Thus, if there are 5000 candidates, we generates [datum x 5001] scores.
'''

import argparse, os, time
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from collections import namedtuple
from utils import load_hparams
from tensorflow.python.ops import lookup_ops
import pickle as pkl

logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model_dir')
  parser.add_argument('-best_model_dir', default='best_eval/')
  parser.add_argument('-candidates', help='All unique training candidates')
  parser.add_argument('-map', help='Candidate indexes, one per txt1 and gt')
  parser.add_argument('-txt1', help='Input sentence, for example conversation context')
  parser.add_argument('-gt', help='Ground Truth candidate')
  parser.add_argument('-candidate_batch_size', default=512, type=int,
                      help='Batch size to compute RNN vectors for Candidates')
  parser.add_argument('-scores_batch_size', default=512, type=int,
                      help='Batch size to compute RNN vectors for Candidates')

  parser.add_argument('-max_len', default=160, type=int,
                      help='Max len of input sentence')

  parser.add_argument('-scores_pkl')


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


class Datum(namedtuple('Datum', 'initializer txt1 len_txt1 txt2 len_txt2 indexes')):
  pass


def create_candidates_with_gt_and_input_iterator(vocab_table, map_file, gt_file, input_file, batch_size, max_len=-1):
  input_dataset = tf.data.TextLineDataset(input_file)
  input_dataset = input_dataset.map(lambda sentence: tf.string_split([sentence]).values)
  input_dataset = input_dataset.map(lambda words: vocab_table.lookup(words))
  if max_len > 0:
    input_dataset = input_dataset.map(lambda words: words[-max_len:])

  gt_dataset = tf.data.TextLineDataset(gt_file)
  gt_dataset = gt_dataset.map(lambda sentence: tf.string_split([sentence]).values)
  gt_dataset = gt_dataset.map(lambda words: vocab_table.lookup(words))

  map_dataset = tf.data.TextLineDataset(map_file)
  map_dataset = map_dataset.map(lambda sentence: tf.string_split([sentence], delimiter=',').values)
  map_dataset = map_dataset.map(lambda words: tf.cast(tf.string_to_number(words), tf.int32))

  dataset = tf.data.Dataset.zip((input_dataset, gt_dataset, map_dataset))
  dataset = dataset.map(lambda txt1, txt2, indexes: (txt1, tf.size(txt1), txt2, tf.size(txt2), indexes))
  dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]),
                                                            tf.TensorShape([None]), tf.TensorShape([]),
                                                            tf.TensorShape([None])))
  iterator = dataset.make_initializable_iterator()

  txt1, len_txt1, txt2, len_txt2, indexes = iterator.get_next()
  return Datum(iterator.initializer, txt1, len_txt1, txt2, len_txt2, indexes)


class EvalModel:
  def __init__(self, hparams, iterator, cv=None):
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

    if cv is not None:
      M = tf.Variable(tf.eye(self.hparams.d), name='M')

      with tf.variable_scope('rnn', reuse=True):
        _, state_txt2 = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt1_vectors,
                                          sequence_length=self.iterator.len_txt1,
                                          dtype=tf.float32)
      self.vec_txt2 = state_txt2.h
      self.saver = tf.train.Saver(tf.global_variables())

      self.WC = tf.get_variable('candidate_vectors', shape=[cv.shape[0], cv.shape[1]])
      self.WC_assign = tf.assign(self.WC, cv)

      self.candidate_vectors = tf.nn.embedding_lookup(self.WC, self.iterator.indexes)

      #Concatenate bs x 1 x d with bs x NC x d; Result bs x NC+1 x d
      self.gt_with_candidate_vectors = tf.concat([tf.reshape(self.vec_txt2, [-1, 1, self.hparams.d]),
                                                  self.candidate_vectors], 1)
      scores = tf.matmul(tf.reshape(tf.matmul(self.vec_txt1, M), [-1, 1, self.hparams.d]),
                              tf.matrix_transpose(self.gt_with_candidate_vectors))

      self.scores = tf.reshape(scores, [tf.shape(self.vec_txt1)[0], -1])
    else:
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


  def compute_scores(self, sess):
    sess.run(self.iterator.initializer)
    start_time = time.time()
    all_scores = []
    num_batches = 0
    while True:
      try:
        all_scores.extend(sess.run(self.scores))
        if num_batches % 100 == 0:
          logging.info('Num_batches: %d Sentences: %d'%(num_batches, len(all_scores)))
      except tf.errors.OutOfRangeError:
        return all_scores, time.time() - start_time


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

def convert_to_numpy_array(list_vectors):
  cv = np.zeros((len(list_vectors), len(list_vectors[0])))
  for index in range(len(list_vectors)):
    cv[index, :] = list_vectors[index]
  return cv


def main():
  args = setup_args()
  logging.info(args)

  #Let us first load hparams for the trained model
  hparams = load_hparams(os.path.join(args.model_dir, 'hparams'))
  logging.info(hparams)

  #Let us create vocab table next
  vocab_table = lookup_ops.index_table_from_file(hparams.vocab, default_value=0)
  list_candidate_vectors = get_candidate_vectors(vocab_table, args, hparams)
  candidate_vectors = convert_to_numpy_array(list_candidate_vectors)
  del list_candidate_vectors

  tf.reset_default_graph()
  vocab_table = lookup_ops.index_table_from_file(hparams.vocab, default_value=0)
  iterator = create_candidates_with_gt_and_input_iterator(vocab_table, args.map, args.gt, args.txt1,
                                                          args.scores_batch_size, args.max_len)

  model = EvalModel(hparams, iterator, candidate_vectors)

  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(model.WC_assign)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join(args.model_dir, args.best_model_dir))
    model.saver.restore(sess, latest_ckpt)
    all_scores, time_taken = model.compute_scores(sess)

  logging.info('Num scores: %d Time: %ds'%(len(all_scores), time_taken))
  with open(args.scores_pkl, 'wb') as fw:
    pkl.dump(all_scores, fw)



if __name__ == '__main__':
  main()