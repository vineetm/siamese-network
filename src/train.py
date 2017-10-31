import argparse
import os
import tensorflow as tf
from tensorflow.contrib.training import HParams

logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
  parser = argparse.ArgumentParser()
  #Data related params
  parser.add_argument('data_dir', help='Data directory')
  parser.add_argument('-txt1', help='txt1 suffix', default='txt1')
  parser.add_argument('-txt2', help='txt1 suffix', default='txt2')

  parser.add_argument('-train', help='train prefix', default='train')
  parser.add_argument('-valid', help='valid prefix', default='valid')

  parser.add_argument('-vocab_suffix', help='vocab suffix', default='vocab.txt')

  parser.add_argument('-vocab_size', default=30000, type=int, help='vocab size')
  parser.add_argument('-d', default=128, type=int, help='vocab size')

  parser.add_argument('-lr', default=1.0, type=float, help='learning rate')
  parser.add_argument('-opt', default='sgd', help='Optimization algo: sgd|adam')

  args = parser.parse_args()
  return args


def build_hparams(args):
  train_txt1 = os.path.join(args.data_dir, '%s.%s'%(args.train, args.txt1))
  train_txt2 = os.path.join(args.data_dir, '%s.%s'%(args.train, args.txt2))

  valid_txt1 = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.txt1))
  valid_txt2 = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.txt2))

  vocab_path = os.path.join(args.data_dir, '%s' % (args.vocab_suffix))

  return HParams(train_txt1 = train_txt1,
                 train_txt2 = train_txt2,

                 valid_txt1 = valid_txt1,
                 valid_txt2 = valid_txt2,

                 vocab_path = vocab_path,
                 vocab_size = args.vocab_size,

                 d = args.d,

                 lr = args.lr,
                 opt = args.opt
                 )


def main():
  args = setup_args()
  hparams = build_hparams(args)
  logging.info(hparams)

if __name__ == '__main__':
  main()