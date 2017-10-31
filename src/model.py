import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.learn import ModeKeys

logging = tf.logging
logging.set_verbosity(logging.INFO)

class SiameseModel:
  def __init__(self, hparams, iterator, mode):
    self.mode = mode
    self.v = hparams.vocab_size
    self.d = hparams.d
    self.iterator = iterator

    self.W = tf.Variable(tf.truncated_normal([self.v, self.d]), name='embeddings')
    self.txt1_vectors = tf.nn.embedding_lookup(self.W, self.iterator.txt1, name='txt1v')
    self.txt2_vectors = tf.nn.embedding_lookup(self.W, self.iterator.txt2, name='txt2v')

    rnn_cell = rnn.BasicLSTMCell(self.d)
    with tf.variable_scope('rnn'):
      outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt1_vectors,
                                         sequence_length=self.iterator.len_txt1, dtype=tf.float32)

    #This is batch_size x d
    self.c = state.h

    with tf.variable_scope('rnn', reuse=True):
      outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt2_vectors,
                                         sequence_length=self.iterator.len_txt2, dtype=tf.float32)

    self.r = state.h
    self.M = tf.Variable(tf.eye(self.d), name='M')

    self.logits = tf.reduce_sum(tf.multiply(self.c, tf.matmul(self.r, self.M)), axis=1)
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

  def train(self, sess):
    assert self.mode == ModeKeys.TRAIN
    return sess.run([self.train_step, self.loss, self.train_summary])