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

  parser.add_argument('-train_batch_size', default=32, type=int, help='Trainign batch size')
  parser.add_argument('-vocab_size', default=30000, type=int, help='vocab size')
  parser.add_argument('-d', default=128, type=int, help='vocab size')

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

                 d = args.d,

                 lr = args.lr,
                 max_norm = args.max_norm,
                 opt = args.opt,

                 model_dir = args.model_dir
                 )


def main():
  args = setup_args()
  hparams = build_hparams(args)
  logging.info(hparams)

  #Create Training graph
  train_graph = tf.Graph()

  with train_graph.as_default():
    vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)
    train_iterator = create_labeled_data_iterator(hparams.train_txt1, hparams.train_txt2, hparams.train_labels,
                                                  vocab_table, hparams.train_batch_size)
    train_model = SiameseModel(hparams, train_iterator, ModeKeys.TRAIN)


  summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'train_log'))

  train_sess = tf.Session(graph=train_graph)
  with train_graph.as_default():
    train_sess.run(tf.global_variables_initializer())
    train_sess.run(tf.tables_initializer())
    train_sess.run(train_iterator.init)

  for step in itertools.count():
    _, loss, train_summary = train_model.train(train_sess)
    summary_writer.add_summary(train_summary, step)
    logging.info('Step: %d Loss: %.2f'%(step, loss))



if __name__ == '__main__':
  main()