import logging, argparse, os, codecs, itertools, time, json
import tensorflow as tf
import numpy as np
from tensorflow.contrib.training import HParams
from tensorflow.python.ops import lookup_ops
from tensorflow.contrib.learn import ModeKeys

from iterator_utils import create_train_dataset_iterator, create_infer_dataset_iterator
from model import RNNPredictor

def setup_args():
  parser = argparse.ArgumentParser()

  #1. Data files

  #This code assumes a parallel corpus of sentences and labels
  #It is okay for context to be absent
  parser.add_argument('-len_max_sentence', default=-1, type=int)
  parser.add_argument('-train_sentences', default=None, help='Train sentences for which vector would be computed')
  parser.add_argument('-train_labels', default=None, help='Train labels file')
  parser.add_argument('-train_context', default=None, help='Additional context')

  parser.add_argument('-valid_sentences', default=None, help='valid sentences for which vector would be computed')
  parser.add_argument('-valid_labels', default=None, help='valid labels file')
  parser.add_argument('-valid_context', default=None, help='Additional context')

  parser.add_argument('-infer_sentences', default=None, help='infer sentences for which vector would be computed')
  parser.add_argument('-infer_context', default=None, help='Additional context')
  parser.add_argument('-infer_out', default=None, help='Additional context')
  parser.add_argument('-prob_cutoff', default=0.5, type=float)

  parser.add_argument('-vocab_input', default=None, help='Vocab file to convert word to ID. line=0 is considered as UNK')
  parser.add_argument('-vocab_output', default=None, help='Vocab file to convert label to ID. line=0 is considered as UNK')

  #2. Model parameters
  parser.add_argument('-d', default=128, type=int, help='Number of units. This is same for word embedding, RNN cell, '
                                                        'class weights')

  parser.add_argument('-pos_scaling', default=10, type=int)

  parser.add_argument('-size_vocab_input',  default=5000, type=int, help='Input vocab size')
  parser.add_argument('-size_vocab_output', default=5000, type=int, help='Ouput vocab size')
  parser.add_argument('-dropout', default=0.0, type=float, help='Dropout for RNN Cell')

  parser.add_argument('-lr', default=0.001, type=float, help='Learning rate')

  parser.add_argument('-max_norm', default=5.0, type=float, help='Learning rate')

  parser.add_argument('-train_batch_size', default=64, type=int)
  parser.add_argument('-valid_batch_size', default=256, type=int)
  parser.add_argument('-infer_batch_size', default=256, type=int)

  #3. Checkpoint related params
  parser.add_argument('-model_dir', default=None, help='Model directory')

  parser.add_argument('-seed', type=int, default=1543)

  parser.add_argument('-steps_per_eval', default=1000, type=int, help='Steps per evaluation')
  parser.add_argument('-steps_per_stats', default=200, type=int, help='Steps per stats and model checkpoint')

  parser.add_argument('-sample_negative_labels', default=False, action='store_true')

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
                 infer_context=args.infer_context,
                 infer_out = args.infer_out,
                 prob_cutoff = args.prob_cutoff,

                 vocab_input = args.vocab_input,
                 vocab_output = args.vocab_output,

                 d = args.d,
                 pos_scaling = args.pos_scaling,
                 size_vocab_input = args.size_vocab_input,
                 size_vocab_output = args.size_vocab_output,
                 dropout = args.dropout,
                 lr = args.lr,
                 max_norm = args.max_norm,

                 train_batch_size = args.train_batch_size,
                 valid_batch_size = args.valid_batch_size,
                 infer_batch_size=args.valid_batch_size,

                 model_dir = args.model_dir,
                 seed = args.seed,

                 steps_per_eval = args.steps_per_eval,
                 steps_per_stats= args.steps_per_stats,
                 len_max_sentence = args.len_max_sentence,

                 sample_negative_labels = args.sample_negative_labels
                 )


def save_hparams(hparams, dir):
  hparams_file = os.path.join(dir, "hparams")
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
  if 'len_max_sentence' in hparams:
    len_max_sentence = hparams.len_max_sentence
  else:
    len_max_sentence = -1

  if 'sample_negative_labels' in hparams and hparams.sample_negative_labels:
    sample_negative_labels = True
  else:
    sample_negative_labels = False

  valid_graph = tf.Graph()
  with valid_graph.as_default():
    tf.set_random_seed(hparams.seed)
    vocab_table_input = lookup_ops.index_table_from_file(hparams.vocab_input, default_value=0)
    vocab_table_output = lookup_ops.index_table_from_file(hparams.vocab_output, default_value=0)


    valid_iterator = create_train_dataset_iterator(hparams.valid_sentences, vocab_table_input, hparams.valid_labels,
                                             vocab_table_output, hparams.size_vocab_output, hparams.valid_batch_size,
                                                   hparams.pos_scaling, len_max_sentence, sample_negative_labels)

    valid_model = RNNPredictor(hparams, valid_iterator, ModeKeys.EVAL)
    valid_sess = tf.Session()

    valid_sess.run(tf.global_variables_initializer())
    valid_sess.run(tf.tables_initializer())

    init_valid_loss, time_taken, _ = valid_model.eval(valid_sess)
    logging.info('Initial Val_loss: %.4f T:%ds' % (init_valid_loss, time_taken))

    if sample_negative_labels is False:
      f1, pr, re, _, _, _, f1_time = valid_model.f1_eval(valid_sess)
      logging.info('Initial F1:%.4f Pr:%.4f Re:%.4f T:%ds'% (f1, pr, re, f1_time))

  # Create Model dir if required
  if not tf.gfile.Exists(hparams.model_dir):
    logging.info('Creating Model dir: %s' % hparams.model_dir)
    tf.gfile.MkDir(hparams.model_dir)
  save_hparams(hparams, hparams.model_dir)

  train_saver_path = os.path.join(hparams.model_dir, 'tr')
  valid_saver_path = os.path.join(hparams.model_dir, 'best_eval')
  valid_f1_saver_path = os.path.join(hparams.model_dir, 'best_f1')
  tf.gfile.MakeDirs(valid_saver_path)
  save_hparams(hparams, valid_saver_path)

  if sample_negative_labels is False:
    tf.gfile.MakeDirs(valid_f1_saver_path)
    save_hparams(hparams, valid_f1_saver_path)

  valid_saver_path = os.path.join(valid_saver_path, 'sm')
  if sample_negative_labels is False:
    valid_f1_saver_path = os.path.join(valid_f1_saver_path, 'sm')

  summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'train_log'))

  # Create training model
  train_graph = tf.Graph()
  with train_graph.as_default():
    tf.set_random_seed(hparams.seed)
    vocab_table_input = lookup_ops.index_table_from_file(hparams.vocab_input, default_value=0)
    vocab_table_output = lookup_ops.index_table_from_file(hparams.vocab_output, default_value=0)

    train_iterator = create_train_dataset_iterator(hparams.train_sentences, vocab_table_input, hparams.train_labels,
                                             vocab_table_output, hparams.size_vocab_output, hparams.train_batch_size,
                                                   hparams.pos_scaling, len_max_sentence, sample_negative_labels)

    train_model = RNNPredictor(hparams, train_iterator, ModeKeys.TRAIN)
    train_sess = tf.Session()

    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_iterator.init)

  #Training Loop
  best_valid_loss = 100.0
  best_f1_score = 0.0
  last_eval_step = 0
  last_stats_step = 0

  epoch_num = 0
  epoch_st_time = time.time()

  for train_step in itertools.count():
    try:
      _, train_loss, train_summary = train_model.train(train_sess)

      if train_step - last_stats_step >= hparams.steps_per_stats:
        logging.info('Epoch: %d Step: %d Train_Loss: %.4f'%(epoch_num, train_step, train_loss))
        train_model.saver.save(train_sess, train_saver_path, train_step)
        summary_writer.add_summary(train_summary, train_step)
        last_stats_step = train_step

      if train_step - last_eval_step >= hparams.steps_per_eval:
        latest_train_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
        valid_model.saver.restore(valid_sess, latest_train_ckpt)

        valid_loss, valid_time_taken, eval_summary = valid_model.eval(valid_sess)
        summary_writer.add_summary(eval_summary, train_step)

        if sample_negative_labels is False:
          f1, pr, re, f1_summary, pr_summary, re_summary, f1_time = valid_model.f1_eval(valid_sess)
          summary_writer.add_summary(f1_summary, train_step)
          summary_writer.add_summary(pr_summary, train_step)
          summary_writer.add_summary(re_summary, train_step)

          if f1 > best_f1_score:
            valid_model.saver.save(valid_sess, valid_f1_saver_path, train_step)
            logging.info('Epoch: %d Step: %d F1 Improved: New: %.4f Old: %.4f T:%ds Pr: %.4f Re: %.4f'%
                         (epoch_num, train_step, f1, best_f1_score, f1_time, pr, re))
            best_f1_score = f1
          else:
            logging.info('Epoch: %d Step: %d F1 Worse: New: %.4f Old: %.4f T:%ds Pr: %.4f Re: %.4f' %
                       (epoch_num, train_step, f1, best_f1_score, f1_time, pr, re))

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


def index_to_word_map(vocab_file):
  vocab_table = {}
  index = 0
  for word in open(vocab_file):
    word = word.strip()
    vocab_table[index] = word
    index +=1
  return vocab_table


def do_infer(hparams, args):
  if 'len_max_sentence' in hparams:
    len_max_sentence = hparams.len_max_sentence
  else:
    len_max_sentence = -1

  infer_graph = tf.Graph()

  rev_vocab_table = index_to_word_map(hparams.vocab_output)
  all_words = np.array([rev_vocab_table[index] for index in range(hparams.size_vocab_output)])

  with infer_graph.as_default():
    vocab_table_input = lookup_ops.index_table_from_file(hparams.vocab_input, default_value=0)

    infer_iterator = create_infer_dataset_iterator(args.infer_sentences, vocab_table_input, args.infer_batch_size, len_max_sentence)

    infer_model = RNNPredictor(hparams, infer_iterator, ModeKeys.INFER)
    infer_sess = tf.Session()
    infer_sess.run(tf.tables_initializer())
    latest_train_ckpt = tf.train.latest_checkpoint(args.model_dir)
    infer_model.saver.restore(infer_sess, latest_train_ckpt)

    fw = open(args.infer_out, 'w')
    start_time = time.time()
    all_probs, num_batches = infer_model.get_all_probs(infer_sess)
    logging.info('Infer time: %ds Batches: %d datums: %d'%((time.time() - start_time), num_batches, len(all_probs)))

  for datum_prob in all_probs:
    fw.write('%s\n'%' '.join(all_words[datum_prob > args.prob_cutoff]))


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
