from tensorflow.contrib.learn import ModeKeys
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import rnn
from collections import OrderedDict

class RNNPredictor:
  def __init__(self, hparams, iterator, mode):
    self.iterator = iterator
    self.mode = mode

    self.hparams = hparams

    self.W = tf.get_variable(name = 'embeddings', shape=[self.hparams.size_vocab_input, self.hparams.d])

    #batch_size x len_sentence x d
    self.word_vectors = tf.nn.embedding_lookup(self.W, self.iterator.sentence)

    with tf.variable_scope('rnn'):
      rnn_cell = rnn.BasicLSTMCell(self.hparams.d)

      #Dropout wrapper at training time
      if mode == ModeKeys.TRAIN and self.hparams.dropout > 0.0:
        rnn_cell = rnn.DropoutWrapper(rnn_cell, input_keep_prob=(1.0 - self.hparams.dropout))

      _ , state = tf.nn.dynamic_rnn(rnn_cell, inputs=self.word_vectors, sequence_length=self.iterator.len_sentence,
                                    dtype=tf.float32)

    #This is batch_size x d
    self.sentence_vector = state.h

    #Add weights for each output label
    self.LW = tf.get_variable(name = 'label_weights', shape=[self.hparams.d, self.hparams.size_vocab_output])

    #In order to save model...
    self.saver = tf.train.Saver(tf.global_variables())

    #This is batch_size x output_vocab
    self.logits = tf.matmul(self.sentence_vector, self.LW)

    #This too is batch_size x output_vocab
    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
      self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.iterator.labels)

      self.batch_loss = self.batch_loss * self.iterator.weights
      #Now, this is a float
      self.loss = tf.reduce_mean(self.batch_loss)

    if mode == ModeKeys.TRAIN:
      self.opt = tf.train.AdamOptimizer(self.hparams.lr)
      #FIXME: Add gradient clipping
      self.train_step = self.opt.minimize(self.loss)

    if mode == ModeKeys.INFER:
      self.probs = tf.sigmoid(self.logits)
      self.num_items = tf.shape(self.iterator.sentence)[0]
      self.cutoff_prob = tf.placeholder(dtype=tf.float32)
      self.pos_labels = tf.squeeze(tf.where(tf.greater_equal(self.probs, self.cutoff_prob)))
      self.lookup_indexes = tf.placeholder(tf.int64)


  def lookup_index(self, sess, rev_vocab_table, index):
    return sess.run(rev_vocab_table.lookup(self.lookup_indexes) , {self.lookup_indexes:index})


  #Returns num_true x 2 items, where index=0 represents datum number
  def get_pos_label_classes(self, infer_sess, rev_vocab_table, min_prob, fw):
    infer_sess.run(self.iterator.init)
    while True:
      try:
        pos_labels, num_items = infer_sess.run([self.pos_labels, self.num_items], {self.cutoff_prob:min_prob})
        output_classes = [[] for _ in range(num_items)]
        for datum_num, label_index in pos_labels:
          output_classes[datum_num].append(self.lookup_index(infer_sess, rev_vocab_table, label_index).decode('utf-8'))
        for datum_num in range(num_items):
          fw.write('%s\n'%' '.join(output_classes[datum_num]))
      except tf.errors.OutOfRangeError:
        return


  def train(self, training_session):
    assert self.mode == ModeKeys.TRAIN
    return training_session.run([self.train_step, self.loss])


  def eval(self, eval_session):
    assert self.mode == ModeKeys.EVAL

    #Eval is just average loss over entire dataset
    start_time = time.time()
    eval_session.run(self.iterator.init)
    eval_losses = []
    while True:
      try:
        eval_losses.append(eval_session.run(self.loss))
      except tf.errors.OutOfRangeError:
        return np.mean(eval_losses), time.time() - start_time