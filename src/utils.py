import tensorflow as tf, codecs, json


def load_hparams(hparams_file):
  if tf.gfile.Exists(hparams_file):
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print("Cannot load hparams file")
        return None
    return hparams
  else:
    return None