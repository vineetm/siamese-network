from tensorflow.contrib.learn import ModeKeys
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import rnn


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

    #This is batch_size x output_vocab
    self.logits = tf.matmul(self.sentence_vector, self.LW)

    #This too is batch_size x output_vocab
    self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.iterator.labels)

    #Now, this is a float
    self.loss = tf.reduce_mean(self.batch_loss)

    if mode == ModeKeys.TRAIN:
      self.opt = tf.train.AdamOptimizer(self.hparams.lr)
      #FIXME: Add gradient clipping
      self.train_step = self.opt.minimize(self.loss)


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