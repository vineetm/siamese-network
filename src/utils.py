import tensorflow as tf, codecs, json
import numpy as np

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

def convert_to_numpy_array(list_vectors):
  cv = np.zeros((len(list_vectors), len(list_vectors[0])))
  for index in range(len(list_vectors)):
    cv[index, :] = list_vectors[index]
  return cv
