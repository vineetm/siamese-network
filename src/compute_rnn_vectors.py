'''
Uses a trained Siamese Network to compute RNN vectors for all sentences in a file
Stores in a pkl file
'''
import argparse, codecs, json, os
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.contrib import rnn
from collections import namedtuple

logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
  parser = argparse.ArgumentParser()
  #Data related params
  parser.add_argument('-model_dir', help='model directory')
  parser.add_argument('-sub_dir', help='model sub directory', default='best_eval/')
  parser.add_argument('-input_sentences', help='Input sentences file')
  parser.add_argument('-output_vectors', help='RNN vector numpy array for each sentence')
  parser.add_argument('-max_len', default=-1, type=int)
  parser.add_argument('-batch_size', default=256)

  args = parser.parse_args()
  return args


def load_hparams(hparams_file):
  if tf.gfile.Exists(hparams_file):
    logging.info("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        logging.info("Can't load hparams file")
        return None
    return hparams
  else:
    return None


class RNNVectorModel:
  def __init__(self, hparams, iterator):
    self.hparams = hparams
    self.iterator = iterator

    #RNN Cell
    # Make default forget gate bias as 2.0, as indicated in paper...
    if 'forget_bias' in hparams:
      rnn_cell = rnn.BasicLSTMCell(self.hparams.d, forget_bias=hparams.forget_bias)
    else:
      rnn_cell = rnn.BasicLSTMCell(self.hparams.d, forget_bias=2.0)

    self.W = tf.get_variable('embeddings', shape=[self.hparams.size_vocab, self.hparams.d])
    self.txt_vectors = tf.nn.embedding_lookup(self.W, self.iterator.txt, name='txtv')

    with tf.variable_scope('rnn'):
      _, state_txt1 = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.txt_vectors, sequence_length=self.iterator.len_txt,
                                         dtype=tf.float32)

    self.vec_txt1 = state_txt1.h
    self.saver = tf.train.Saver(tf.global_variables())

  def compute_rnn_vectors(self, sess):
    sess.run(self.iterator.init)
    try:
      rnn_vectors = sess.run(self.vec_txt1)
      logging.info('Num %d '%len(rnn_vectors))
    except tf.errors.OutOfRangeError:
      return


class BatchedDatum(namedtuple('BatchedDatum', 'txt len_txt init')):
  pass


def create_single_sentence_iterator(file_name, vocab_table, batch_size, max_len):
  dataset = tf.data.TextLineDataset(file_name)
  dataset = dataset.map(lambda sentence: tf.string_split([sentence]).values)
  dataset = dataset.map(lambda words: vocab_table.lookup(words))
  if max_len > 0:
    dataset = dataset.map(lambda words: words[-max_len:])

  dataset = dataset.map(lambda words: (tf.cast(words, tf.int32), tf.size(words)))
  dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
  iterator = dataset.make_initializable_iterator()
  txt, len_txt = iterator.get_next()
  return BatchedDatum(txt, len_txt, iterator.initializer)


def main():
  args = setup_args()
  logging.info(args)

  #Load saved hparams from model directory
  hparams_file = os.path.join(args.model_dir, 'hparams')
  hparams = load_hparams(hparams_file)
  logging.info(hparams)

  vocab_table = lookup_ops.index_table_from_file(hparams.vocab, default_value=0)
  iterator = create_single_sentence_iterator(args.input_sentences, vocab_table, args.batch_size, args.max_len)
  model = RNNVectorModel(hparams, iterator)

  sess = tf.Session()
  sess.run(tf.tables_initializer())

  latest_ckpt = tf.train.latest_checkpoint(os.path.join(args.model_dir, args.sub_dir))
  model.saver.restore(sess, latest_ckpt)
  model.compute_rnn_vectors(sess)


if __name__ == '__main__':
  main()


