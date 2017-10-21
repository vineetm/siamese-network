import tensorflow as tf
from tensorflow.contrib import data
from tensorflow.python.ops import lookup_ops

from collections import namedtuple
NUM_TEXT2 = 10

logging = tf.logging
logging.set_verbosity(logging.INFO)

'''
Convert a TextLineDataset to an correponding word indexes

1. Split a string by space (assumes string to be already tokenized)
2. Use vocab_table to assign word index

Note we work on .values for output generated using tf string_split
'''
def get_word_index_dataset(path, vocab_table):
  dataset = data.TextLineDataset(path)

  # Split words
  dataset = dataset.map(lambda line: (tf.string_split([line]).values))

  # Convert to word indexes using vocab_table
  dataset = dataset.map(lambda words: (vocab_table.lookup(words)))

  return dataset


class BatchedInput(namedtuple('BatchedInput', 'text1 text2 labels init')):
  pass

def create_train_iterator(text1_path, text2_path, labels_path, batch_size, vocab_table):
  text1_dataset = get_word_index_dataset(text1_path, vocab_table)
  text2_dataset = get_word_index_dataset(text2_path, vocab_table)

  labels_dataset = data.TextLineDataset(labels_path)
  labels_dataset = labels_dataset.map(lambda string: tf.string_to_number(string))

  dataset = data.Dataset.zip((text1_dataset, text2_dataset, labels_dataset))
  dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))

  iterator = dataset.make_initializable_iterator()
  text1, text2, label = iterator.get_next()

  return BatchedInput(text1, text2, label, iterator.initializer)


class BatchedValidInput(namedtuple('BatchedValidInput', 'text1 text2 init')):
  pass


def create_valid_iterator(text1_path, text2_path, batch_size, vocab_table):
  #Create dataset for text1
  text1_dataset = get_word_index_dataset(text1_path, vocab_table)

  text2_datasets = []
  for pnum in range(NUM_TEXT2):
    text2_datasets.append(get_word_index_dataset('%s.p%d'%(text2_path, pnum), vocab_table))

  datasets_list = []
  datasets_list.append(text1_dataset)
  datasets_list.extend(text2_datasets)

  dataset = data.Dataset.zip(tuple(datasets_list))
  padded_shapes_list = [tf.TensorShape([None]) for _ in range(len(datasets_list))]
  dataset = dataset.padded_batch(batch_size, padded_shapes=tuple(padded_shapes_list))

  iterator = dataset.make_initializable_iterator()

  datum = iterator.get_next()

  return BatchedValidInput(datum[0], datum[1:], iterator.initializer)


def test_valid_iterator():
  text1_path = 'data/valid.txt1'
  text2_path = 'data/valid.txt2.p%d'
  vocab_table = lookup_ops.index_table_from_file('data/vocab100k.txt', default_value=0)

  iterator = create_valid_iterator(text1_path, text2_path, 2, vocab_table)
  sess = tf.Session()
  sess.run(tf.tables_initializer())
  sess.run(iterator.init)

  datum = sess.run(iterator)

if __name__ == '__main__':
  test_valid_iterator()

