import logging, argparse, os, codecs
import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.ops import lookup_ops

from iterator_utils import create_dataset_iterator

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

  #3. Checkpoint related params
  parser.add_argument('-model_dir', default=None, help='Model directory')

  parser.add_argument('-seed', type=int, default=1543)
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

                 vocab_input = args.vocab_input,
                 vocab_output = args.vocab_output,

                 d = args.d,
                 size_vocab_input = args.size_vocab_input,
                 size_vocab_output = args.size_vocab_output,
                 dropout = args.dropout,
                 lr = args.lr,

                 train_batch_size = args.train_batch_size,
                 valid_batch_size = args.valid_batch_size,

                 model_dir = args.model_dir,
                 seed = args.seed
                 )


def save_hparams(hparams):
  hparams_file = os.path.join(hparams.model_dir, "hparams")
  logging.info("Saving hparams to %s" % hparams_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())


def main():
  args = setup_args()
  logging.info(args)
  hparams = build_hparams(args)

  # Create Model dir if required
  if not tf.gfile.Exists(hparams.model_dir):
    logging.info('Creating Model dir: %s' % hparams.model_dir)
    tf.gfile.MkDir(hparams.model_dir)
  save_hparams(hparams)


  # Create Training graph, and session
  train_graph = tf.Graph()
  with train_graph.as_default():
    tf.set_random_seed(hparams.seed)
    vocab_table_input = lookup_ops.index_table_from_file(hparams.vocab_input, default_value=0)
    vocab_table_output = lookup_ops.index_table_from_file(hparams.vocab_output, default_value=0)

    train_iterator = create_dataset_iterator(hparams.train_sentences, vocab_table_input, hparams.train_labels, vocab_table_output,
                                             hparams.size_vocab_output, hparams.batch_size)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()
