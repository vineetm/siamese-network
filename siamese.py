import tensorflow as tf
import tensorflow.contrib as contrib
from collections import namedtuple
import argparse, os, codecs

from tensorflow.python.ops import lookup_ops
from tensorflow.contrib import data

logging = tf.logging
logging.set_verbosity(logging.INFO)

HPARAMS = 'hparams.json'

def setup_args():
  parser = argparse.ArgumentParser()

  #Data parameters
  parser.add_argument('-data_dir', default='.', help='Data directory')
  parser.add_argument('-vocab_suffix', default='vocab.txt', help='Vocab file suffix')

  #Training data parameters
  parser.add_argument('-train_prefix', default='train', help='Train file prefix')
  parser.add_argument('-text1', default='text1', help='Text1 suffix')
  parser.add_argument('-text2', default='text2', help='Text2 suffix')
  parser.add_argument('-labels', default='labels', help='Labels')

  #Model parameters
  parser.add_argument('-d', default=300, type=int, help='#Units')
  parser.add_argument('-vocab', default=100000, type=int, help='Vocab size')
  parser.add_argument('-lr', default=1.0, type=float, help='Learning rate')
  parser.add_argument('-train_batch_size', default=128, type=int, help='Train batch Size')

  #Checkpoint parameters
  parser.add_argument('-out_dir', default='out', help='Directory to save model checkpoints')

  args = parser.parse_args()
  return args


def create_hparams(flags):
  return contrib.training.HParams(
    data_dir = flags.data_dir,
    vocab_path = os.path.join(flags.data_dir, flags.vocab_suffix),
    text1_path = os.path.join(flags.data_dir, '%s.%s'%(flags.train_prefix, flags.text1)),
    text2_path = os.path.join(flags.data_dir, '%s.%s' % (flags.train_prefix, flags.text2)),
    labels_path= os.path.join(flags.data_dir, '%s.%s' % (flags.train_prefix, flags.labels)),

    d = flags.d,
    vocab = flags.vocab,
    lr = flags.lr,
    train_batch_size = flags.train_batch_size,

    out_dir = flags.out_dir
  )


def save_hparams(hparams):
  if not tf.gfile.Exists(hparams.out_dir):
    logging.info('Creating out dir: %s'%hparams.out_dir)
    tf.gfile.MakeDirs(hparams.out_dir)

  hparams_file = os.path.join(hparams.out_dir, HPARAMS)
  logging.info('Saving hparams: %s'%hparams_file)
  with codecs.getwriter('utf-8')(tf.gfile.GFile(hparams_file, 'wb')) as f:
    f.write(hparams.to_json())

class BatchedInput(namedtuple('BatchedInput', 'text1 text2 label init')):
  pass

def create_train_iterator(text1_path, text2_path, labels_path, batch_size, vocab_table):
  def convert_to_word_index(path):
    dataset = data.TextLineDataset(path)

    #Split words
    dataset = dataset.map(lambda line: (tf.string_split([line]).values))

    #Convert to word indexes using vocab_table
    dataset = dataset.map(lambda words: (vocab_table.lookup(words)))

    return dataset

  text1_dataset = convert_to_word_index(text1_path)
  text2_dataset = convert_to_word_index(text2_path)

  labels_dataset = data.TextLineDataset(labels_path)
  labels_dataset = labels_dataset.map(lambda string: tf.string_to_number(string))

  dataset = data.Dataset.zip((text1_dataset, text2_dataset, labels_dataset))
  dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))

  iterator = dataset.make_initializable_iterator()
  text1, text2, label = iterator.get_next()

  return BatchedInput(text1, text2, label, iterator.initializer)



class SiameseModel:
  def __init__(self, hparams, mode):
    self.mode = mode
    self.d = hparams.d
    self.vocab = hparams.vocab

    self.vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)

    #Setup iterator
    if mode == contrib.learn.ModeKeys.TRAIN:
      self.iterator = create_train_iterator(hparams.text1_path, hparams.text2_path, hparams.labels_path,
                                              hparams.train_batch_size, self.vocab_table)

    self.batch_size = tf.shape(self.iterator.text1)[0]
    self.W = tf.get_variable('embeddings', [self.vocab, self.d])





def main():
  FLAGS = setup_args()
  hparams = create_hparams(FLAGS)
  save_hparams(hparams)

  train_graph = tf.Graph()

  with train_graph.as_default():
    train_model = SiameseModel(hparams, contrib.learn.ModeKeys.TRAIN)
    train_sess = tf.Session()
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_model.iterator.init)
    train_sess.run(tf.global_variables_initializer())

    while True:
      try:
        bs = train_sess.run(train_model.batch_size)
        logging.info('bs: %d'%bs)
      except tf.errors.OutOfRangeError:
        logging.info('End of epoch!')
        break


if __name__ == '__main__':
  main()