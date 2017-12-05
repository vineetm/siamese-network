import argparse
import os, time
import codecs

import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.ops import lookup_ops
from tensorflow.contrib.learn import ModeKeys

logging = tf.logging
logging.set_verbosity(logging.INFO)

from iterator_utils import create_labeled_data_iterator, create_labeled_data_iterator_with_context
from model import SiameseModel

import itertools

def setup_args():
  parser = argparse.ArgumentParser()
  #Data related params
  parser.add_argument('-model_dir', help='model directory')

  parser.add_argument('-train_txt1', help='Train txt1', default=None)
  parser.add_argument('-train_txt2', help='Train txt2', default=None)
  parser.add_argument('-train_labels', help='Train labels', default=None)
  parser.add_argument('-train_context', help='Train context', default=None)

  parser.add_argument('-valid_txt1', help='Valid txt1', default=None)
  parser.add_argument('-valid_txt2', help='Valid txt2', default=None)
  parser.add_argument('-valid_labels', help='Valid labels', default=None)
  parser.add_argument('-valid_context', help='Valid labels', default=None)

  parser.add_argument('-vocab', help='vocab suffix', default='vocab.txt')

  parser.add_argument('-size_train_batch', default=128, type=int, help='Train batch size')
  parser.add_argument('-size_valid_batch', default=128, type=int, help='Valid batch size')

  parser.add_argument('-size_vocab', default=30000, type=int, help='vocab size')
  parser.add_argument('-d', default=128, type=int, help='Number of units')

  parser.add_argument('-steps_per_eval', default=1000, type=int, help='Steps per eval')
  parser.add_argument('-steps_per_stats', default=200, type=int, help='Steps per stats')

  parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
  parser.add_argument('-max_norm', default=5.0, type=float, help='learning rate')
  parser.add_argument('-opt', default='adam', help='Optimization algo: sgd|adam')

  parser.add_argument('-dropout', default=0.0, type=float, help='Dropout, only applied at training')

  parser.add_argument('-seed', default=1543, type=int)
  parser.add_argument('-forget_bias', default=1.0, type=float)

  args = parser.parse_args()
  return args


def build_hparams(args):
  return HParams(train_txt1 = args.train_txt1,
                 train_txt2 = args.train_txt2,
                 train_labels = args.train_labels,
                 train_context = args.train_context,

                 valid_txt1 = args.valid_txt1,
                 valid_txt2 = args.valid_txt2,
                 valid_labels = args.valid_labels,
                 valid_context = args.valid_context,

                 vocab = args.vocab,
                 size_vocab = args.size_vocab,

                 size_train_batch = args.size_train_batch,
                 size_valid_batch = args.size_valid_batch,

                 d = args.d,
                 steps_per_eval = args.steps_per_eval,
                 steps_per_stats = args.steps_per_stats,

                 lr = args.lr,
                 max_norm = args.max_norm,
                 opt = args.opt,
                 dropout = args.dropout,

                 model_dir = args.model_dir,
                 seed = args.seed,
                 forget_bias = args.forget_bias,
                 )


def save_hparams(hparams):
  hparams_file = os.path.join(hparams.model_dir, "hparams")
  logging.info("Saving hparams to %s" % hparams_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())


def main():
  args = setup_args()
  hparams = build_hparams(args)
  logging.info(hparams)

  #Create Valid graph, and session
  valid_graph = tf.Graph()

  with valid_graph.as_default():
    # Set random seed
    tf.set_random_seed(args.seed)
    vocab_table = lookup_ops.index_table_from_file(hparams.vocab, default_value=0)
    if hparams.train_context:
      valid_iterator = create_labeled_data_iterator_with_context(hparams.valid_context, hparams.valid_txt1, hparams.valid_txt2, hparams.valid_labels,
                                                    vocab_table, hparams.size_valid_batch)
    else:

      valid_iterator = create_labeled_data_iterator(hparams.valid_txt1, hparams.valid_txt2, hparams.valid_labels,
                                                  vocab_table, hparams.size_valid_batch)

    valid_model = SiameseModel(hparams, valid_iterator, ModeKeys.EVAL)

    #Create Training session and init its variables, tables and iterator.
    valid_sess = tf.Session()
    valid_sess.run(valid_iterator.init)

    valid_sess.run(tf.global_variables_initializer())
    valid_sess.run(tf.tables_initializer())

    eval_loss, time_taken, _ = valid_model.eval(valid_sess)
    logging.info('Init Val Loss: %.4f Time: %ds'%(eval_loss, time_taken))

  #Create Model dir if required
  if not tf.gfile.Exists(hparams.model_dir):
    logging.info('Creating Model dir: %s'%hparams.model_dir)
    tf.gfile.MkDir(hparams.model_dir)
  save_hparams(hparams)

  #Create Training graph, and session
  train_graph = tf.Graph()

  with train_graph.as_default():
    # Set random seed
    tf.set_random_seed(args.seed)

    #First word in vocab file is UNK (see prep_data/create_vocab.py)
    vocab_table = lookup_ops.index_table_from_file(hparams.vocab, default_value=0)

    if hparams.train_context:
      train_iterator = create_labeled_data_iterator_with_context(hparams.train_context, hparams.train_txt1,
                                                                 hparams.train_txt2, hparams.train_labels,
                                                                 vocab_table, hparams.size_train_batch)
    else:
      train_iterator = create_labeled_data_iterator(hparams.train_txt1, hparams.train_txt2, hparams.train_labels,
                                                    vocab_table, hparams.size_train_batch)

    train_model = SiameseModel(hparams, train_iterator, ModeKeys.TRAIN)

    #Create Training session and init its variables, tables and iterator.
    train_sess = tf.Session()
    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_iterator.init)

  #Training loop
  summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'train_log'))
  epoch_num = 0
  epoch_start_time = time.time()
  best_eval_loss = 100.0

  #When did we last check validation data
  last_eval_step = 0

  #When did we last save training stats and checkoiint
  last_stats_step = 0


  train_saver_path = os.path.join(hparams.model_dir, 'sm')
  valid_saver_path = os.path.join(hparams.model_dir, 'best_eval')
  tf.gfile.MakeDirs(valid_saver_path)
  valid_saver_path = os.path.join(valid_saver_path, 'sm')

  for step in itertools.count():
    try:
      _, loss, train_summary = train_model.train(train_sess)

      #Steps per stats
      if step - last_stats_step >= hparams.steps_per_stats:
        logging.info('Epoch: %d Step %d: Train_Loss: %.4f'%(epoch_num, step, loss))
        train_model.saver.save(train_sess, train_saver_path, step)
        summary_writer.add_summary(train_summary, step)
        last_stats_step = step

      # Eval model and print stats
      if step - last_eval_step >= hparams.steps_per_eval:
        latest_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
        valid_model.saver.restore(valid_sess, latest_ckpt)
        eval_loss, time_taken, eval_summary = valid_model.eval(valid_sess)
        summary_writer.add_summary(eval_summary, step)

        if eval_loss < best_eval_loss:
          valid_model.saver.save(valid_sess, valid_saver_path, step)
          logging.info('Epoch: %d Step: %d Valid_Loss Improved New: %.4f Old: %.4f'%(epoch_num, step, eval_loss, best_eval_loss))
          best_eval_loss = eval_loss
        else:
          logging.info('Epoch: %d Step: %d Valid_Loss Worse New: %.4f Old: %.4f'%(epoch_num, step, eval_loss, best_eval_loss))
        last_eval_step = step

    except tf.errors.OutOfRangeError:
      logging.info('Epoch %d END Time: %ds'%(epoch_num, time.time() - epoch_start_time))
      epoch_num += 1

      with train_graph.as_default():
        train_sess.run(train_iterator.init)
      epoch_start_time = time.time()


if __name__ == '__main__':
  main()