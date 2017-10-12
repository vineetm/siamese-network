import tensorflow as tf
from dataset_iterator import create_dataset_iterator
from tensorflow.python.ops import lookup_ops
import os

class SiameseNetwork:
  def __init__(self, d, V, batch_size, seed=1543):
    tf.set_random_seed(seed)

    #Define word embedding matrix
    self.embeddings = tf.get_variable('embeddings', [V, d])

    #Define placeholders for context, utterance and label
    self.context = tf.placeholder(tf.int32, shape=[batch_size, None])
    self.utterance = tf.placeholder(tf.int32, shape=[batch_size, None])
    self.label = tf.placeholder(tf.float32, shape=[batch_size])

    #Convert context and utterance to vectors, shape = bs x T x d
    context_vector = tf.nn.embedding_lookup(self.embeddings, self.context)

    utterance_vector = tf.nn.embedding_lookup(self.embeddings, self.utterance)


    #Define RNN
    with tf.variable_scope('rnn') as scope:
      rnn_cell = tf.contrib.rnn.BasicLSTMCell(d)
      outputs, batch_context = tf.nn.dynamic_rnn(rnn_cell, context_vector, dtype=tf.float32)

    with tf.variable_scope('rnn', reuse=True) as scope:
      outputs, batch_utterance = tf.nn.dynamic_rnn(rnn_cell, utterance_vector, dtype=tf.float32)

    M = tf.get_variable('M', [d, d])
    logits = tf.reduce_sum(tf.multiply(batch_context.h, tf.matmul(batch_utterance.h, M)), axis=1)

    batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.label, logits = logits)

    self.loss = tf.reduce_mean(batch_loss)

  def train(self, sess, iterator, learning_rate=1.0):
    params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, params)
    clipped_gradients = tf.clip_by_global_norm(gradients, 5.0)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    while True:
      ctx_batch, utterance_batch, labels_batch = sess.run(iterator.get_next())
      loss, _ = sess.run([self.loss, update_step  ], feed_dict={self.context: ctx_batch,
                                                           self.utterance: utterance_batch, self.label: labels_batch})
      print('Loss: %f'%loss)


def main():
  OUT_DATA_PATH = 'data'
  train_ctx = os.path.join(OUT_DATA_PATH, 'train.ctx')
  train_utterance = os.path.join(OUT_DATA_PATH, 'train.utterance')
  train_label = os.path.join(OUT_DATA_PATH, 'train.label')
  vocab_file = os.path.join(OUT_DATA_PATH, 'vocab100k.txt')

  vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value=0)
  batch_size = 32
  V = 100000
  d = 300

  sn = SiameseNetwork(d, V, batch_size)
  train_iterator = create_dataset_iterator(train_ctx, train_utterance, train_label, vocab_table, batch_size)

  sess = tf.Session()
  sess.run(tf.tables_initializer())
  sess.run(tf.global_variables_initializer())
  sess.run(train_iterator.initializer)

  sn.train(sess, train_iterator, learning_rate=0.1)


if __name__ == '__main__':
    main()


