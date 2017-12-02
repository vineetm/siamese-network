from tensorflow.contrib.learn import ModeKeys
import tensorflow as tf
import numpy as np
import time, logging
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

      #Now, let us clip gradients as we are dealing with RNN
      params = tf.trainable_variables()
      logging.info('Trainable params: %s'%params)

      gradients = tf.gradients(self.loss, params)
      clipped_gradients, self.grad_norm = tf.clip_by_global_norm(gradients, hparams.max_norm)

      self.train_step = self.opt.apply_gradients(zip(clipped_gradients, params))
      self.train_summary = tf.summary.merge([tf.summary.scalar('train_loss', self.loss),
                                             tf.summary.scalar('grad_norm', self.grad_norm)])

    if mode == ModeKeys.INFER or ModeKeys.EVAL:
      self.probs = tf.sigmoid(self.logits)
      self.num_items = tf.shape(self.iterator.sentence)[0]
      self.cutoff_prob = tf.placeholder(dtype=tf.float32, shape=())
      self.pos_labels = tf.squeeze(tf.where(tf.greater_equal(self.probs, self.cutoff_prob)))

      self.lookup_indexes = tf.placeholder(tf.int64)

    if mode == ModeKeys.EVAL:
      with tf.variable_scope('metrics') as scope:
        self.precision = tf.metrics.precision_at_thresholds(labels=self.iterator.labels, predictions=self.probs,
                                                           thresholds=[0.5])
        self.recall = tf.metrics.recall_at_thresholds(labels=self.iterator.labels, predictions=self.probs,
                                                           thresholds=[0.5])

        vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
      self.reset_metrics_op = tf.variables_initializer(vars)

  def lookup_index(self, sess, rev_vocab_table, index):
    return sess.run(rev_vocab_table.lookup(self.lookup_indexes) , {self.lookup_indexes:index})

  def get_all_probs(self, infer_sess):
    infer_sess.run(self.iterator.init)
    all_probs = []
    num_batches = 0
    while True:
      try:
        all_probs.extend(infer_sess.run(self.probs))
        num_batches += 1
        logging.info('Completed %d'%num_batches)
      except tf.errors.OutOfRangeError:
        return all_probs, num_batches


  #Returns num_true x 2 items, where index=0 represents datum number
  def old_get_pos_label_classes(self, infer_sess, rev_vocab_table, min_prob, fw):
    infer_sess.run(self.iterator.init)
    num_batches = 0
    while True:
      try:
        pos_labels, num_items = infer_sess.run([self.pos_labels, self.num_items], {self.cutoff_prob:min_prob})
        logging.info('Batch %d done'%num_batches)
        num_batches += 1
        output_classes = [[] for _ in range(num_items)]
        for datum_num, label_index in pos_labels:
          output_classes[datum_num].append(self.lookup_index(infer_sess, rev_vocab_table, label_index).decode('utf-8'))
        for datum_num in range(num_items):
          fw.write('%s\n'%' '.join(output_classes[datum_num]))
      except tf.errors.OutOfRangeError:
        return num_batches


  def train(self, training_session):
    assert self.mode == ModeKeys.TRAIN
    return training_session.run([self.train_step, self.loss, self.train_summary])


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
        eval_loss = np.mean(eval_losses)
        eval_summary = tf.Summary(value=[tf.Summary.Value(tag='eval_loss', simple_value=eval_loss)])
        return eval_loss, time.time() - start_time, eval_summary


  #Compute pr, recall and F1 scores for positive labels
  def f1_eval(self, eval_session):
    #Init the dataset iterator
    start_time = time.time()
    eval_session.run(self.reset_metrics_op)
    eval_session.run(self.iterator.init)
    while True:
      try:
        pr, re = eval_session.run([self.precision, self.recall])
      except tf.errors.OutOfRangeError:
        f1 = (2 * pr[1] * re[1]) / (pr[1] + re[1])
        logging.info('F1: %.4f Pr: %.4f Re: %.4f'%(f1, pr[1], re[1]))
        f1_summary = tf.Summary(value=[tf.Summary.Value(tag='f1', simple_value=float(f1))])
        pr_summary = tf.Summary(value=[tf.Summary.Value(tag='pr', simple_value=float(pr[1]))])
        re_summary = tf.Summary(value=[tf.Summary.Value(tag='re', simple_value=float(re[1]))])
        return f1, pr[1], re[1], f1_summary, pr_summary, re_summary, time.time() - start_time