import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.learn import ModeKeys

import time

logging = tf.logging
logging.set_verbosity(logging.INFO)


class SiameseModel:
  def __init__(self, hparams, iterator, mode):
    #Mode specifies Train, Eval or Infer
    self.mode = mode

    self.V = hparams.vocab_size

    self.d = hparams.d
    self.num_units = hparams.num_units
    self.iterator = iterator

    self.W = tf.get_variable('embeddings', shape=[self.V, self.d])
    #self.W = tf.Variable(tf.random_uniform(shape=[self.V, self.d], minval=-0.25, maxval=0.25), name='embeddings')

    self.txt1_vectors = tf.nn.embedding_lookup(self.W, self.iterator.txt1, name='txt1v')
    self.txt2_vectors = tf.nn.embedding_lookup(self.W, self.iterator.txt2, name='txt2v')


    #Make default forget gate bias as 2.0, as indicated in paper...
    if 'forget_bias' in hparams:
      rnn_cell_wd = rnn.BasicLSTMCell(self.num_units, forget_bias=hparams.forget_bias)
    else:
      rnn_cell_wd = rnn.BasicLSTMCell(self.num_units, forget_bias=2.0)

    # Dropout is only applied at train. Not required at test as the inputs are scaled accordingly
    if mode == ModeKeys.TRAIN:
      rnn_cell = rnn.DropoutWrapper(rnn_cell_wd, input_keep_prob=(1 - hparams.dropout),
                                    output_keep_prob=(1 - hparams.dropout))

      logging.info('Dropout: %.2f'%hparams.dropout)
    else:
      rnn_cell = rnn_cell_wd

    with tf.variable_scope('rnn'):
      _, state_txt1 = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt1_vectors, sequence_length=self.iterator.len_txt1,
                                         dtype=tf.float32)

    #This is batch_size x num_units
    self.vec_txt1 = state_txt1.h

    if 'use_context' in hparams and hparams.use_context:
      logging.info('Using Separate ctx vector')
      self.context_vectors = tf.nn.embedding_lookup(self.W, self.iterator.context, name='ctxv')

      #RNN to compute ctx vector
      with tf.variable_scope('rnn', reuse=True):
        _, state_ctx = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt1_vectors, sequence_length=self.iterator.len_txt1, dtype=tf.float32)
      vec_ctx = state_ctx.h

      #RNN to combine vec_txt1 and vec_ctx
      with tf.variable_scope('rnn', reuse=True):
        _, level1  = tf.nn.dynamic_rnn(cell=rnn_cell_wd, inputs=tf.stack([vec_ctx, self.vec_txt1], axis=1),
                                sequence_length=tf.tile([2], [tf.size(self.iterator.len_txt1)]), dtype=tf.float32)

      vec_txt1 = level1.h
      self.vec_txt1 = vec_txt1

    with tf.variable_scope('rnn', reuse=True):
      outputs_txt2, state_txt2 = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt2_vectors,
                                           sequence_length=self.iterator.len_txt2, dtype=tf.float32)

    self.vec_txt2 = state_txt2.h
    self.M = tf.Variable(tf.eye(self.num_units), name='M')

    self.logits = tf.reduce_sum(tf.multiply(self.vec_txt1, tf.matmul(self.vec_txt2, self.M)), axis=1)

    self.saver = tf.train.Saver(tf.global_variables())

    if self.mode == ModeKeys.TRAIN or self.mode == ModeKeys.EVAL:
      self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iterator.labels, logits=self.logits)
      self.loss = tf.reduce_mean(self.batch_loss)

    #We only need optimizer while training
    if self.mode == ModeKeys.TRAIN:
      if hparams.opt == 'sgd':
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=hparams.lr)
      elif hparams.opt == 'adam':
        self.opt = tf.train.AdamOptimizer(learning_rate=hparams.lr)
      else:
        self.opt = None
        logging.error('Bad optimization param!')

      #Now, let us clip gradients as we are dealing with RNN
      params = tf.trainable_variables()
      logging.info('Trainable params: %s'%params)

      gradients = tf.gradients(self.loss, params)
      clipped_gradients, self.grad_norm = tf.clip_by_global_norm(gradients, hparams.max_norm)
      self.train_step = self.opt.apply_gradients(zip(clipped_gradients, params))
      self.train_summary = tf.summary.merge([tf.summary.scalar('train_loss', self.loss),
                                             tf.summary.scalar('grad_norm', self.grad_norm)])


  def compute_scores(self, sess, out_file, freq=100):
    assert self.mode == ModeKeys.INFER
    # Initialize iterator
    sess.run(self.iterator.init)

    num_batches = 0
    fw = open(out_file, 'w')
    while True:
      try:
        logits = sess.run(self.logits)
        for logit in logits:
          fw.write('%.4f\n'%logit)

        num_batches += 1
        if num_batches % freq == 0:
          logging.info('Batches Completed: %d'%num_batches)

      except tf.errors.OutOfRangeError:
        fw.close()
        return


  def train(self, sess):
    assert self.mode == ModeKeys.TRAIN
    return sess.run([self.train_step, self.loss, self.train_summary])


  def eval(self, sess):
    assert self.mode == ModeKeys.EVAL

    #Initialize iterator
    sess.run(self.iterator.init)

    start_time = time.time()
    total_loss = 0.0
    num_batches = 0
    while True:
      try:
        total_loss += sess.run(self.loss)
        num_batches += 1
      except tf.errors.OutOfRangeError:
        avg_loss = total_loss / num_batches
        eval_summary = tf.Summary(value=[tf.Summary.Value(tag='valid_loss', simple_value=avg_loss)])
        return avg_loss, time.time() - start_time, eval_summary