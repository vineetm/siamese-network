import logging, argparse, os, codecs, itertools, time, json
import tensorflow as tf
import numpy as np
from tensorflow.contrib.training import HParams
from tensorflow.python.ops import lookup_ops
from tensorflow.contrib.learn import ModeKeys

from iterator_utils import create_dataset_iterator
from model import RNNPredictor

def setup_args():
  parser = argparse.ArgumentParser()

  #1. Data files

  #This code assumes a parallel corpus of sentences and labels
  #It is okay for context to be absent
  parser.add_argument('-train_sentences', default=None, help='Train sentences for which vector would be computed')
  parser.add_argument('-train_labels', default=None, help='Train labels file')
  parser.add_argument('-train_context', default=None, help='Additional context')

  parser.add_argument('-valid_sentences', default=None, help='valid sentences for which vector would be computed')
  parser.add_argument('-valid_labels', default=None, help='valid labels file')
  parser.add_argument('-valid_context', default=None, help='Additional context')

  parser.add_argument('-infer_sentences', default=None, help='infer sentences for which vector would be computed')
  parser.add_argument('-infer_labels', default=None, help='infer labels file')
  parser.add_argument('-infer_context', default=None, help='Additional context')
  parser.add_argument('-infer_out', default=None, help='Additional context')

  parser.add_argument('-vocab_input', default=None, help='Vocab file to convert word to ID. line=0 is considered as UNK')
  parser.add_argument('-vocab_output', default=None, help='Vocab file to convert label to ID. line=0 is considered as UNK')

  #2. Model parameters
  parser.add_argument('-d', default=128, type=int, help='Number of units. This is same for word embedding, RNN cell, '
                                                        'class weights')

  parser.add_argument('-size_vocab_input',  default=5000, type=int, help='Input vocab size')
  parser.add_argument('-size_vocab_output', default=5000, type=int, help='Ouput vocab size')
  parser.add_argument('-dropout', default=0.0, type=float, help='Dropout for RNN Cell')

  parser.add_argument('-lr', default=0.001, type=float, help='Learning rate')

  parser.add_argument('-train_batch_size', default=64, type=int)
  parser.add_argument('-valid_batch_size', default=256, type=int)
  parser.add_argument('-infer_batch_size', default=16, type=int)

  #3. Checkpoint related params
  parser.add_argument('-model_dir', default=None, help='Model directory')

  parser.add_argument('-seed', type=int, default=1543)

  parser.add_argument('-steps_per_eval', default=1000, type=int, help='Steps per evaluation')
  parser.add_argument('-steps_per_stats', default=200, type=int, help='Steps per stats and model checkpoint')

  args = parser.parse_args()

  return args

#FIXME: There ought to be a better way then to repeat params
def build_hparams(args):
  return HParams(train_sentences = args.train_sentences,
                 train_labels = args.train_labels,
                 train_context = args.train_context,

                 valid_sentences=args.valid_sentences,
                 valid_labels=args.valid_labels,
                 valid_context=args.valid_context,

                 infer_sentences=args.infer_sentences,
                 infer_labels=args.infer_labels,
                 infer_context=args.infer_context,
                 infer_out = args.infer_out,

                 vocab_input = args.vocab_input,
                 vocab_output = args.vocab_output,

                 d = args.d,
                 size_vocab_input = args.size_vocab_input,
                 size_vocab_output = args.size_vocab_output,
                 dropout = args.dropout,
                 lr = args.lr,

                 train_batch_size = args.train_batch_size,
                 valid_batch_size = args.valid_batch_size,
                 infer_batch_size=args.valid_batch_size,

                 model_dir = args.model_dir,
                 seed = args.seed,

                 steps_per_eval = args.steps_per_eval,
                 steps_per_stats= args.steps_per_stats
                 )


def save_hparams(hparams):
  hparams_file = os.path.join(hparams.model_dir, "hparams")
  logging.info("Saving hparams to %s" % hparams_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())

def load_hparams(hparams_file):
  if tf.gfile.Exists(hparams_file):
    logging.info("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        logging.info("  can't load hparams file")
        return None
    return hparams
  else:
    return None


def do_train(hparams):
  # Create validation graph, and session
  valid_graph = tf.Graph()
  with valid_graph.as_default():
    tf.set_random_seed(hparams.seed)
    vocab_table_input = lookup_ops.index_table_from_file(hparams.vocab_input, default_value=0)
    vocab_table_output = lookup_ops.index_table_from_file(hparams.vocab_output, default_value=0)

    train_iterator = create_dataset_iterator(hparams.valid_sentences, vocab_table_input, hparams.valid_labels,
                                             vocab_table_output,
                                             hparams.size_vocab_output, hparams.valid_batch_size)

    valid_model = RNNPredictor(hparams, train_iterator, ModeKeys.EVAL)
    valid_sess = tf.Session()

    valid_sess.run(tf.global_variables_initializer())
    valid_sess.run(tf.tables_initializer())

    init_valid_loss, time_taken = valid_model.eval(valid_sess)
    logging.info('Initial Val_loss: %.4f T:%ds' % (init_valid_loss, time_taken))
  # Create Model dir if required
  if not tf.gfile.Exists(hparams.model_dir):
    logging.info('Creating Model dir: %s' % hparams.model_dir)
    tf.gfile.MkDir(hparams.model_dir)
  save_hparams(hparams)

  train_saver_path = os.path.join(hparams.model_dir, 'tr')
  valid_saver_path = os.path.join(hparams.model_dir, 'best_eval')
  tf.gfile.MakeDirs(valid_saver_path)
  valid_saver_path = os.path.join(valid_saver_path, 'sm')

  # Create training model
  train_graph = tf.Graph()
  with train_graph.as_default():
    tf.set_random_seed(hparams.seed)
    vocab_table_input = lookup_ops.index_table_from_file(hparams.vocab_input, default_value=0)
    vocab_table_output = lookup_ops.index_table_from_file(hparams.vocab_output, default_value=0)

    train_iterator = create_dataset_iterator(hparams.train_sentences, vocab_table_input, hparams.train_labels,
                                             vocab_table_output,
                                             hparams.size_vocab_output, hparams.train_batch_size)

    train_model = RNNPredictor(hparams, train_iterator, ModeKeys.TRAIN)
    train_sess = tf.Session()

    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_iterator.init)

  #Training Loop
  best_valid_loss = 100.0
  last_eval_step = 0
  last_stats_step = 0

  epoch_num = 0
  epoch_st_time = time.time()

  for train_step in itertools.count():
    try:
      _, train_loss = train_model.train(train_sess)

      if train_step - last_stats_step >= hparams.steps_per_stats:
        logging.info('Epoch: %d Step: %d Train_Loss: %.4f'%(epoch_num, train_step, train_loss))
        train_model.saver.save(train_sess, train_saver_path, train_step)
        last_stats_step = train_step

      if train_step - last_eval_step >= hparams.steps_per_eval:
        latest_train_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
        valid_model.saver.restore(valid_sess, latest_train_ckpt)

        valid_loss, valid_time_taken = valid_model.eval(valid_sess)
        if valid_loss < best_valid_loss:
          valid_model.saver.save(valid_sess, valid_saver_path, train_step)
          logging.info('Epoch: %d Step: %d Valid Loss Improved: New: %.4f Old: %.4f T:%ds'%(epoch_num, train_step,
                                                                                      valid_loss, best_valid_loss, valid_time_taken))
          best_valid_loss = valid_loss
        else:
          logging.info('Epoch: %d Step: %d Valid Loss Worse: New: %.4f Old: %.4f T:%ds' % (epoch_num, train_step,
                                                                                     valid_loss, best_valid_loss, valid_time_taken))
        last_eval_step = train_step

    except tf.errors.OutOfRangeError:
      epoch_num += 1
      logging.info('Epoch %d DONE T:%ds Step: %d'%(epoch_num, time.time() - epoch_st_time, train_step))
      train_sess.run(train_iterator.init)
      epoch_st_time = time.time()


def do_infer(hparams, args):
  infer_graph = tf.Graph()

  with infer_graph.as_default():
    vocab_table_input = lookup_ops.index_table_from_file(hparams.vocab_input, default_value=0)
    vocab_table_output = lookup_ops.index_table_from_file(hparams.vocab_output, default_value=0)
    rev_vocab_table_output = lookup_ops.index_to_string_table_from_file(hparams.vocab_output)

    infer_iterator = create_dataset_iterator(args.infer_sentences, vocab_table_input, args.infer_labels,
                                             vocab_table_output,
                                             hparams.size_vocab_output, args.infer_batch_size)

    infer_model = RNNPredictor(hparams, infer_iterator, ModeKeys.INFER)
    infer_sess = tf.Session()
    infer_sess.run(tf.tables_initializer())
    latest_train_ckpt = tf.train.latest_checkpoint(args.model_dir)
    infer_model.saver.restore(infer_sess, latest_train_ckpt)

    fw = open(args.infer_out, 'w')
    infer_model.get_pos_label_classes(infer_sess, rev_vocab_table_output, 0.1, fw)


def main():
  args = setup_args()
  logging.info(args)

  if args.train_sentences is None:
    logging.info('Infer Only mode')
    hparams_file = os.path.join(args.model_dir, 'hparams')
    hparams = load_hparams(hparams_file)
    logging.info(hparams)
    do_infer(hparams, args)
  else:
    hparams = build_hparams(args)
    logging.info('Train mode')
    do_train(hparams)



if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()
