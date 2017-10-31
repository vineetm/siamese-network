import argparse
import os

import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.ops import lookup_ops
from tensorflow.contrib.learn import ModeKeys

logging = tf.logging
logging.set_verbosity(logging.INFO)

from iterator_utils import create_labeled_data_iterator
from model import SiameseModel

import itertools

def setup_args():
  parser = argparse.ArgumentParser()
  #Data related params
  parser.add_argument('data_dir', help='Data directory')
  parser.add_argument('model_dir', help='model directory')

  parser.add_argument('-txt1', help='txt1 suffix', default='txt1')
  parser.add_argument('-txt2', help='txt1 suffix', default='txt2')
  parser.add_argument('-labels', help='txt1 suffix', default='labels')

  parser.add_argument('-train', help='train prefix', default='train')
  parser.add_argument('-valid', help='valid prefix', default='valid')

  parser.add_argument('-vocab_suffix', help='vocab suffix', default='vocab.txt')

  parser.add_argument('-train_batch_size', default=128, type=int, help='Train batch size')
  parser.add_argument('-valid_batch_size', default=128, type=int, help='Valid batch size')

  parser.add_argument('-vocab_size', default=30000, type=int, help='vocab size')
  parser.add_argument('-d', default=128, type=int, help='embedding size')

  parser.add_argument('-steps_per_eval', default=100, type=int, help='Steps per eval')
  parser.add_argument('-steps_per_ckpt', default=50, type=int, help='Steps per ckpt')

  parser.add_argument('-lr', default=1.0, type=float, help='learning rate')
  parser.add_argument('-max_norm', default=5.0, type=float, help='learning rate')
  parser.add_argument('-opt', default='sgd', help='Optimization algo: sgd|adam')

  args = parser.parse_args()
  return args


def build_hparams(args):
  train_txt1 = os.path.join(args.data_dir, '%s.%s'%(args.train, args.txt1))
  train_txt2 = os.path.join(args.data_dir, '%s.%s'%(args.train, args.txt2))
  train_labels = os.path.join(args.data_dir, '%s.%s' % (args.train, args.labels))

  valid_txt1 = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.txt1))
  valid_txt2 = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.txt2))
  valid_labels = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.labels))

  vocab_path = os.path.join(args.data_dir, '%s' % (args.vocab_suffix))

  return HParams(train_txt1 = train_txt1,
                 train_txt2 = train_txt2,
                 train_labels = train_labels,

                 valid_txt1 = valid_txt1,
                 valid_txt2 = valid_txt2,
                 valid_labels = valid_labels,

                 vocab_path = vocab_path,
                 vocab_size = args.vocab_size,

                 train_batch_size = args.train_batch_size,
                 valid_batch_size=args.valid_batch_size,

                 d = args.d,
                 steps_per_eval = args.steps_per_eval,
                 steps_per_ckpt = args.steps_per_ckpt,

                 lr = args.lr,
                 max_norm = args.max_norm,
                 opt = args.opt,

                 model_dir = args.model_dir
                 )


def main():
  args = setup_args()
  hparams = build_hparams(args)
  logging.info(hparams)

  #Create Training graph, and session
  train_graph = tf.Graph()

  with train_graph.as_default():
    vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)
    train_iterator = create_labeled_data_iterator(hparams.train_txt1, hparams.train_txt2, hparams.train_labels,
                                                  vocab_table, hparams.train_batch_size)
    train_model = SiameseModel(hparams, train_iterator, ModeKeys.TRAIN)

    #Create Training session and init its variables, tables and iterator.
    train_sess = tf.Session()
    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_iterator.init)

  #Create Training graph, and session
  valid_graph = tf.Graph()

  with valid_graph.as_default():
    vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)
    valid_iterator = create_labeled_data_iterator(hparams.valid_txt1, hparams.valid_txt2, hparams.valid_labels,
                                                  vocab_table, hparams.valid_batch_size)
    valid_model = SiameseModel(hparams, valid_iterator, ModeKeys.EVAL)

    #Create Training session and init its variables, tables and iterator.
    valid_sess = tf.Session()
    valid_sess.run(valid_iterator.init)

  #Initial Evaluation
  with valid_graph.as_default():
    valid_sess.run(tf.global_variables_initializer())
    valid_sess.run(tf.tables_initializer())

    eval_loss, time_taken = valid_model.eval(valid_sess)
    logging.info('Init Val Loss: %.4f Time: %ds'%(eval_loss, time_taken))

  #Training loop
  summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'train_log'))
  last_eval_step = 0
  last_ckpt_step = 0

  train_saver_path = os.path.join(hparams.model_dir, 'sm')
  for step in itertools.count():
    _, loss, train_summary = train_model.train(train_sess)
    summary_writer.add_summary(train_summary, step)

    #Steps per ckpt, save train model
    if step - last_ckpt_step >= hparams.steps_per_ckpt:
      train_model.saver.save(train_sess, train_saver_path, step)
      logging.info('Step %d: Saved Train model'%step)
      last_ckpt_step = step

    # Eval model and print stats
    if step - last_eval_step >= hparams.steps_per_eval:
      logging.info('Step %d: Train_Loss: %.4f'%(step, loss))

      #Load last saved model from checkpoint
      latest_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
      valid_model.saver.restore(valid_sess, latest_ckpt)
      eval_loss, time_taken = valid_model.eval(valid_sess)
      logging.info('Step %d: Val_Loss: %.4f Time: %ds' % (step, eval_loss, time_taken))

      last_eval_step = step


if __name__ == '__main__':
  main()