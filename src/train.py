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
  parser.add_argument('data_dir', help='Data directory')
  parser.add_argument('model_dir', help='model directory')

  parser.add_argument('-txt1', help='txt1 suffix', default='txt1')
  parser.add_argument('-txt2', help='txt2 suffix', default='txt2')
  parser.add_argument('-context', help= 'context', default='context')
  parser.add_argument('-labels', help='txt1 suffix', default='labels')

  parser.add_argument('-train', help='train prefix', default='train')
  parser.add_argument('-valid', help='valid prefix', default='valid')

  parser.add_argument('-vocab_suffix', help='vocab suffix', default='vocab.txt')

  parser.add_argument('-train_batch_size', default=128, type=int, help='Train batch size')
  parser.add_argument('-valid_batch_size', default=128, type=int, help='Valid batch size')

  parser.add_argument('-vocab_size', default=30000, type=int, help='vocab size')
  parser.add_argument('-d', default=128, type=int, help='word embedding size')
  parser.add_argument('-num_units', default=128, type=int, help='RNN num units')

  parser.add_argument('-steps_per_train_summary', default=20, type=int, help='Steps per train summary')
  parser.add_argument('-steps_per_eval', default=1000, type=int, help='Steps per eval')
  parser.add_argument('-steps_per_stats', default=100, type=int, help='Steps per stats')

  parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
  parser.add_argument('-max_norm', default=5.0, type=float, help='learning rate')
  parser.add_argument('-opt', default='adam', help='Optimization algo: sgd|adam')

  parser.add_argument('-dropout', default=0.0, type=float, help='Dropout, only applied at training')

  parser.add_argument('-diff_rnn', default=False, action='store_true',
                      help='Use different RNN for txt1 and txt2')

  parser.add_argument('-seed', default=1543, type=int)
  parser.add_argument('-forget_bias', default=1.0, type=float)
  parser.add_argument('-use_context', default=False, action='store_true')
  
  parser.add_argument('-shuffle', help='Randomly shuffle dataset', action='store_true', default=False)
  args = parser.parse_args()
  return args


def build_hparams(args):
  train_txt1 = os.path.join(args.data_dir, '%s.%s'%(args.train, args.txt1))
  train_txt2 = os.path.join(args.data_dir, '%s.%s'%(args.train, args.txt2))
  train_labels = os.path.join(args.data_dir, '%s.%s' % (args.train, args.labels))

  if args.use_context:
    train_context = os.path.join(args.data_dir, '%s.%s' % (args.train, args.context))
  else:
    train_context = None

  valid_txt1 = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.txt1))
  valid_txt2 = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.txt2))
  valid_labels = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.labels))

  if args.use_context:
    valid_context = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.context))
  else:
    valid_context = None

  vocab_path = os.path.join(args.data_dir, '%s' % (args.vocab_suffix))

  return HParams(train_txt1 = train_txt1,
                 train_txt2 = train_txt2,
                 train_labels = train_labels,
                 train_context = train_context,

                 valid_txt1 = valid_txt1,
                 valid_txt2 = valid_txt2,
                 valid_labels = valid_labels,
                 valid_context = valid_context,

                 vocab_path = vocab_path,
                 vocab_size = args.vocab_size,

                 train_batch_size = args.train_batch_size,
                 valid_batch_size=args.valid_batch_size,

                 d = args.d,
                 num_units = args.num_units,
                 steps_per_train_summary = args.steps_per_train_summary,
                 steps_per_eval = args.steps_per_eval,
                 steps_per_stats = args.steps_per_stats,

                 lr = args.lr,
                 max_norm = args.max_norm,
                 opt = args.opt,
                 dropout = args.dropout,

                 model_dir = args.model_dir,

                 diff_rnn = args.diff_rnn,

                 shuffle = args.shuffle,
                 seed = args.seed,
                 forget_bias = args.forget_bias,
                 use_context = args.use_context
                 )


def save_hparams(hparams):
  hparams_file = os.path.join(hparams.model_dir, "hparams")
  logging.info("Saving hparams to %s" % hparams_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())


def main():
  args = setup_args()
  hparams = build_hparams(args)

  #Save hparams
  logging.info(hparams)

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

    vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)

    if hparams.use_context:
      train_iterator = create_labeled_data_iterator_with_context(hparams.train_context, hparams.train_txt1, hparams.train_txt2, hparams.train_labels,
                                                  vocab_table, hparams.train_batch_size, hparams.shuffle, hparams.seed)
    else:
      train_iterator = create_labeled_data_iterator(hparams.train_txt1, hparams.train_txt2, hparams.train_labels,
                                                  vocab_table, hparams.train_batch_size, hparams.shuffle, hparams.seed)
    train_model = SiameseModel(hparams, train_iterator, ModeKeys.TRAIN)

    #Create Training session and init its variables, tables and iterator.
    train_sess = tf.Session()
    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_iterator.init)

  #Create Training graph, and session
  valid_graph = tf.Graph()

  with valid_graph.as_default():
    # Set random seed
    tf.set_random_seed(args.seed)
    vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)
    if hparams.use_context:
      valid_iterator = create_labeled_data_iterator_with_context(hparams.valid_context, hparams.valid_txt1, hparams.valid_txt2, hparams.valid_labels,
                                                    vocab_table, hparams.valid_batch_size)
    else:

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

    eval_loss, time_taken, _ = valid_model.eval(valid_sess)
    logging.info('Init Val Loss: %.4f Time: %ds'%(eval_loss, time_taken))

  #Training loop
  summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'train_log'))
  last_summary_step = 0
  last_eval_step = 0
  last_stats_step = 0
  epoch_num = 0
  epoch_start_time = time.time()
  best_eval_loss = 100.0

  train_saver_path = os.path.join(hparams.model_dir, 'sm')
  valid_saver_path = os.path.join(hparams.model_dir, 'best_eval')
  tf.gfile.MakeDirs(valid_saver_path)

  valid_saver_path = os.path.join(valid_saver_path, 'sm')

  for step in itertools.count():
    try:
      _, loss, train_summary = train_model.train(train_sess)


      #Steps per stats
      if step - last_stats_step >= hparams.steps_per_stats:
        logging.info('Step %d: Train_Loss: %.4f'%(step, loss))
        last_stats_step = step

      if step - last_summary_step >= hparams.steps_per_train_summary:
        summary_writer.add_summary(train_summary, step)
        last_summary_step = step

      # Eval model and print stats
      if step - last_eval_step >= hparams.steps_per_eval:
        logging.info('Step %d: Saved Model'%step)
        train_model.saver.save(train_sess, train_saver_path, step)

        #Load last saved model from checkpoint
        load_st_time = time.time()
        latest_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
        valid_model.saver.restore(valid_sess, latest_ckpt)
        logging.info('Step: %d Restore valid Time: %ds'%(step, time.time()-load_st_time))

        eval_loss, time_taken, eval_summary = valid_model.eval(valid_sess)
        logging.info('Step %d: Val_Loss: %.4f Time: %ds' % (step, eval_loss, time_taken))
        summary_writer.add_summary(eval_summary, step)

        if eval_loss < best_eval_loss:
          valid_model.saver.save(valid_sess, valid_saver_path, step)
          logging.info('Step: %d New: %.4f Old: %.4f saved eval'%(step, eval_loss, best_eval_loss))
          best_eval_loss = eval_loss
        else:
          logging.info('Step: %d New: %.4f Old: %.4f Not_saved eval'%(step, eval_loss, best_eval_loss))

        last_eval_step = step

    except tf.errors.OutOfRangeError:
      logging.info('Epoch %d END Time: %ds'%(epoch_num, time.time() - epoch_start_time))
      epoch_num += 1

      with train_graph.as_default():
        train_sess.run(train_iterator.init)

      epoch_start_time = time.time()


if __name__ == '__main__':
  main()