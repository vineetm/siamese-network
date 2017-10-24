import numpy as np
import math
import argparse, os, codecs, itertools, time

import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.python.ops import lookup_ops
from dataset_utils import create_valid_iterator, create_train_iterator


logging = tf.logging
logging.set_verbosity(logging.INFO)

HPARAMS = 'hparams.json'

#Retrieval related constants
RK = [1, 2, 5]

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-seed', default=1543, type=int)
  parser.add_argument('-word2vec', default=None, help='Pre-trained word embeddings')

  #Data parameters
  parser.add_argument('-data_dir', default='.', help='Data directory')
  parser.add_argument('-vocab_suffix', default='vocab.txt', help='Vocab file suffix')

  #Training data parameters
  parser.add_argument('-opt', default='sgd', help='Optimizer: sgd|adam')
  parser.add_argument('-train_prefix', default='train', help='Train file prefix')
  parser.add_argument('-valid_prefix', default='valid', help='Valid file prefix')
  parser.add_argument('-text1', default='txt1', help='Text1 suffix')
  parser.add_argument('-text2', default='txt2', help='Text2 suffix')
  parser.add_argument('-labels', default='labels', help='Labels')

  #Model parameters
  parser.add_argument('-d', default=300, type=int, help='#Units')
  parser.add_argument('-vocab', default=100000, type=int, help='Vocab size')
  parser.add_argument('-lr', default=1.0, type=float, help='Learning rate')
  parser.add_argument('-train_batch_size', default=128, type=int, help='Train batch Size')
  parser.add_argument('-valid_batch_size', default=128, type=int, help='Train batch Size')

  #Checkpoint parameters
  parser.add_argument('-out_dir', default='out', help='Directory to save model checkpoints')
  parser.add_argument('-steps_per_stats', default=100, type=int, help='Steps after which to display stats')
  parser.add_argument('-steps_per_eval', default=100, type=int, help='Steps after which to display stats')

  args = parser.parse_args()
  return args


def create_hparams(flags):
  return contrib.training.HParams(
    word2vec = flags.word2vec,

    opt = flags.opt,
    data_dir = flags.data_dir,
    vocab_path = os.path.join(flags.data_dir, flags.vocab_suffix),
    text1_path = os.path.join(flags.data_dir, '%s.%s'%(flags.train_prefix, flags.text1)),
    text2_path = os.path.join(flags.data_dir, '%s.%s' % (flags.train_prefix, flags.text2)),
    labels_path= os.path.join(flags.data_dir, '%s.%s' % (flags.train_prefix, flags.labels)),

    valid_text1_path = os.path.join(flags.data_dir, '%s.%s'%(flags.valid_prefix, flags.text1)),
    valid_text2_path=os.path.join(flags.data_dir, '%s.%s' % (flags.valid_prefix, flags.text2)),

    d = flags.d,
    vocab = flags.vocab,
    lr = flags.lr,
    train_batch_size = flags.train_batch_size,
    valid_batch_size=flags.valid_batch_size,

    out_dir = flags.out_dir,
    steps_per_stats = flags.steps_per_stats,
    steps_per_eval =  flags.steps_per_eval
  )


def save_hparams(hparams):
  if not tf.gfile.Exists(hparams.out_dir):
    logging.info('Creating out dir: %s'%hparams.out_dir)
    tf.gfile.MakeDirs(hparams.out_dir)

  hparams_file = os.path.join(hparams.out_dir, HPARAMS)
  logging.info('Saving hparams: %s'%hparams_file)
  with codecs.getwriter('utf-8')(tf.gfile.GFile(hparams_file, 'wb')) as f:
    f.write(hparams.to_json())


class SiameseModel:
  def __init__(self, hparams, mode):
    if mode == contrib.learn.ModeKeys.TRAIN:
      logging.info('Creating Training Model')
    else:
      logging.info('Creating Valid Model')

    #Specify Train or Validation model
    self.mode = mode

    #Units, Vocab size
    self.d = hparams.d
    self.vocab = hparams.vocab

    #Create a new graph
    self.graph = tf.Graph()

    with self.graph.as_default():
      self.vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)
      self.reverse_vocab_table = lookup_ops.index_to_string_table_from_file(hparams.vocab_path)

      #Setup iterator
      if mode == contrib.learn.ModeKeys.TRAIN:
        self.iterator = create_train_iterator(hparams.text1_path, hparams.text2_path, hparams.labels_path,
                                              hparams.train_batch_size, self.vocab_table)
      else:
        self.iterator = create_valid_iterator(hparams.valid_text1_path, hparams.valid_text2_path,
                                              hparams.valid_batch_size, self.vocab_table)

      #Batch size is dynamic, only different for the last batch...
      self.batch_size = tf.shape(self.iterator.text1)[0]
      self.create_embeddings(hparams)

      text1_vectors = tf.nn.embedding_lookup(self.W, self.iterator.text1)
      rnn_cell = contrib.rnn.BasicLSTMCell(self.d)
      with tf.variable_scope('rnn'):
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, text1_vectors, dtype=tf.float32)
        t1 = state.h
      self.M = tf.Variable(tf.eye(self.d))

      if mode == contrib.learn.ModeKeys.TRAIN:
        text2_vectors = tf.nn.embedding_lookup(self.W, self.iterator.text2)
        with tf.variable_scope('rnn', reuse=True):
          outputs, state = tf.nn.dynamic_rnn(rnn_cell, text2_vectors, dtype=tf.float32)
          t2 = state.h

        logits = tf.reduce_sum(tf.multiply(t1, tf.matmul(t2, self.M)), axis=1)
        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iterator.labels, logits=logits)
        self.loss = tf.reduce_mean(batch_loss)

        if hparams.opt == 'sgd':
          logging.info('Using sgd optimizer')
          optimizer = tf.train.GradientDescentOptimizer(hparams.lr)
        elif hparams.opt == 'adam':
          logging.info('Using adam optimizer')
          optimizer = tf.train.AdamOptimizer(hparams.lr)
        else:
          logging.error('Do not recognize the optimizer: %s'%hparams.opt)
          return

        self.train_step = optimizer.minimize(self.loss)

        self.train_summary = tf.summary.scalar("train_loss", self.loss)

      elif mode == contrib.learn.ModeKeys.EVAL:
        all_logits = []
        #FIXME: This is ugly
        for i in range(10):
          text2_vectors = tf.nn.embedding_lookup(self.W, self.iterator.text2[i])
          with tf.variable_scope('rnn', reuse=True):
            outputs, state = tf.nn.dynamic_rnn(rnn_cell, text2_vectors, dtype=tf.float32)
            t2 = state.h
          logits = tf.reduce_sum(tf.multiply(t1, tf.matmul(t2, self.M)), axis=1)
          all_logits.append(logits)
        self.all_logits = all_logits

      self.saver = tf.train.Saver(tf.global_variables())


  def create_embeddings(self, hparams):
    # Word Embedding business!
    if hparams.word2vec is None:
      # self.W = tf.get_variable(name='embeddings', shape=[hparams.vocab, hparams.d])
      self.W = tf.truncated_normal(shape=[hparams.vocab, hparams.d])
      logging.info('Fresh embeddings!')
    else:
      W_np = np.load(hparams.word2vec)
      self.W = tf.Variable(name='embeddings', initial_value=W_np)
      logging.info('Init embeddings from %s' % hparams.word2vec)


  def train(self, sess):
    assert self.mode == contrib.learn.ModeKeys.TRAIN
    return sess.run([self.train_step, self.loss, self.train_summary])


  def eval(self, sess, step, summary_writer):

    def calculate_recall(total_correct, total):
      recall = {}
      for k in RK:
        recall[k] = total_correct[k] / total
        summary = tf.Summary(value=[tf.Summary.Value(tag='R@%d'%k, simple_value=recall[k])])
        summary_writer.add_summary(summary, step)
      return recall

    assert self.mode == contrib.learn.ModeKeys.EVAL
    total = 0.0
    # batch_num = 0

    total_correct = {}
    for k in RK:
      total_correct[k] = 0.0

    start_time = time.time()
    logging.info('Step: %d Evaluation START'%step)
    while True:
      try:
        all_logits, batch_size = sess.run([self.all_logits, self.batch_size])
        total += batch_size
        sorted_indexes = np.argsort(all_logits, axis=0)

        for k in RK:
          batch_correct = np.sum(sorted_indexes[:k][:] == 0)
          total_correct[k] += batch_correct
        # batch_num += 1

      except tf.errors.OutOfRangeError:
        logging.info('Step: %d Evaluation END'%step)
        logging.info('Step: %d Correct: %s Total: %d'%(step, total_correct, total))

        recall = calculate_recall(total_correct, total)
        recall_str = ' '.join(['R@%d=%.2f'%(k, recall[k]) for k in RK])
        logging.info('Step: %d Recall: %s Time: %.2fs' %(step, recall_str, (time.time() - start_time)))
        return int(total_correct[1])

  def __str__(self):
    logging.info('Graph: %s'%self.graph)

#Setup session with latest model
def load_saved_model(model, sess, model_dir, msg):
  with model.graph.as_default():
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
      model.saver.restore(sess, latest_ckpt)
      logging.info('%s: Restored saved model: %s' %(msg, latest_ckpt))
    else:
      logging.info('%s: Fresh model!'%msg)
      sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(model.iterator.init)


def main():
  FLAGS = setup_args()
  hparams = create_hparams(FLAGS)
  logging.info(hparams)
  save_hparams(hparams)
  tf.set_random_seed(FLAGS.seed)

  # Setup valid model and session
  valid_model = SiameseModel(hparams, contrib.learn.ModeKeys.EVAL)
  valid_sess = tf.Session(graph=valid_model.graph)
  logging.info('Created Valid Model')


  #Create directory for saving best valid model
  valid_model_dir = os.path.join(hparams.out_dir, 'best_valid')
  logging.info('Create Valid model directory: %s'%valid_model_dir)
  tf.gfile.MakeDirs(valid_model_dir)

  #Setup train model and session
  train_model = SiameseModel(hparams, contrib.learn.ModeKeys.TRAIN)
  train_sess = tf.Session(graph=train_model.graph)
  load_saved_model(train_model, train_sess, hparams.out_dir, "train")
  logging.info('Created Train Model')

  #Training Loop
  last_stats_step = 0
  last_eval_step = 0
  step_time = 0.0
  epoch_num = 0

  best_eval_score = 0

  #Setup summary writer
  summary_writer = tf.summary.FileWriter(os.path.join(hparams.out_dir, 'train_log'), train_model.graph)

  for step in itertools.count():
    try:
      start_time = time.time()
      _, loss, train_summary = train_model.train(train_sess)

      if math.isinf(loss) or math.isnan(loss):
        logging.error('Loss Nan/Inf: %f'%loss)
        return
      step_time += (time.time() - start_time)

      #Write training loss summary
      summary_writer.add_summary(train_summary, step)

      #Time to print stats?
      if step - last_stats_step == hparams.steps_per_stats:
        logging.info('Step: %d loss: %f AvgTime: %.2fs'%(step, loss, step_time/hparams.steps_per_stats))
        step_time = 0.0
        last_stats_step = step

      #Time to evaluate model?
      if step - last_eval_step == hparams.steps_per_eval:
        train_model.saver.save(train_sess, os.path.join(hparams.out_dir, 'siamese.ckpt'), global_step=step)
        logging.info('Saved Training Model at step: %d'%step)

        #Perform eval on saved model
        load_saved_model(valid_model, valid_sess, hparams.out_dir, "eval")
        current_eval = valid_model.eval(valid_sess, step, summary_writer)
        if current_eval > best_eval_score:
          valid_model.saver.save(valid_sess, os.path.join(valid_model_dir, 'siamese.ckpt'), global_step=step)
          logging.info('Step:%d New_Score: %d Old_Score: %d Saved model'%(step, current_eval, best_eval_score))
          best_eval_score = current_eval
        else:
          logging.info('Step:%d New_Score: %d Old_Score: %d Not saved!'%(step, current_eval, best_eval_score))

        last_eval_step = step

    except tf.errors.OutOfRangeError:
      logging.info('Epoch: %d Done'%epoch_num)
      # Perform eval on saved model
      load_saved_model(valid_model, valid_sess, hparams.out_dir, "eval")
      current_eval = valid_model.eval(valid_sess, step)
      if current_eval > best_eval_score:
        valid_model.saver.save(valid_sess, os.path.join(valid_model_dir, 'siamese.ckpt'), global_step=step)
        logging.info('Step:%d New_Score: %d Old_Score: %d Saved model' % (step, current_eval, best_eval_score))
        best_eval_score = current_eval
      else:
        logging.info('Step:%d New_Score: %d Old_Score: %d Not saved!' % (step, current_eval, best_eval_score))
      last_eval_step = step

      epoch_num += 1

      step_time = 0.0
      train_sess.run(train_model.iterator.init)

if __name__ == '__main__':
  main()