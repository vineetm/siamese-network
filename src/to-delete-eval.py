from model import SiameseModel
import tensorflow as tf
import argparse, codecs, json, os

from tensorflow.python.ops import lookup_ops
from tensorflow.contrib.learn import ModeKeys
from iterator_utils import create_data_iterator, create_data_iterator_with_ctx

logging = tf.logging
logging.set_verbosity(logging.INFO)


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model_dir')
  parser.add_argument('-txt1', default=None)
  parser.add_argument('-txt2', default=None)
  parser.add_argument('-ctx', default=None)
  parser.add_argument('-out', default='valid.scores')
  parser.add_argument('-batch_size', default=256, type=int)
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
        logging.info("  can't load hparams file")
        return None
    return hparams
  else:
    return None


def main():
  args = setup_args()
  logging.info(args)

  #Load Hparams
  hparams_file = os.path.join(args.model_dir, 'hparams')
  hparams = load_hparams(hparams_file)
  logging.info(hparams)

  vocab_table = lookup_ops.index_table_from_file(hparams.vocab_path, default_value=0)

  if 'use_context' in hparams and hparams.use_context:
    logging.info('Eval: using context iterator')
    iterator = create_data_iterator_with_ctx(args.ctx, args.txt1, args.txt2, vocab_table, args.batch_size)
  else:
    logging.info('Eval: txt1 and txt2 iterator')
    iterator = create_data_iterator(args.txt1, args.txt2, vocab_table, args.batch_size)
  infer_model = SiameseModel(hparams, iterator, ModeKeys.INFER)

  out_file = os.path.join(args.model_dir, args.out)
  with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(os.path.join(args.model_dir, 'best_eval/'))
    infer_model.saver.restore(sess, latest_ckpt)

    infer_model.compute_scores(sess, out_file, freq=100)


if __name__ == '__main__':
  main()