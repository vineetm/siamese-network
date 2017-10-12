import tensorflow as tf
import os
from tensorflow.contrib import data
from tensorflow.python.ops import lookup_ops

def create_dataset_iterator(ctx_fname, utterance_fname, label_fname, vocab_table, batch_size):
  #Set up the context line dataset
  ctx_dataset = data.TextLineDataset(ctx_fname)

  # Set up the utterance line dataset
  utterance_dataset = data.TextLineDataset(utterance_fname)

  #Setup the labels dataset
  label_dataset = data.TextLineDataset(label_fname)

  #This would split the line into words. Note we need `values`
  ctx_dataset = ctx_dataset.map(lambda line: tf.string_split([line]).values)
  utterance_dataset = utterance_dataset.map(lambda line: tf.string_split([line]).values)

  #Now, let us convert the individual words to integers
  ctx_dataset = ctx_dataset.map(lambda words: (vocab_table.lookup(words)))
  utterance_dataset = utterance_dataset.map(lambda words: (vocab_table.lookup(words)))

  #Cast to int32
  ctx_dataset = ctx_dataset.map(lambda word_indexes: (tf.cast(word_indexes, tf.int32)))
  utterance_dataset = utterance_dataset.map(lambda word_indexes: (tf.cast(word_indexes, tf.int32)))

  #Convert label to float
  label_dataset = label_dataset.map(lambda label: (tf.string_to_number(label)))

  joint_dataset = data.Dataset.zip((ctx_dataset, utterance_dataset, label_dataset))

  joint_dataset = joint_dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]),
                                                        tf.TensorShape([])))

  iterator = joint_dataset.make_initializable_iterator()
  return iterator


def test():
  OUT_DATA_PATH = 'data'
  train_ctx = os.path.join(OUT_DATA_PATH, 'train.ctx')
  train_utterance = os.path.join(OUT_DATA_PATH, 'train.utterance')
  train_label = os.path.join(OUT_DATA_PATH, 'train.label')
  vocab_file = os.path.join(OUT_DATA_PATH, 'vocab100k.txt')

  vocab_table = lookup_ops.index_table_from_file(vocab_file, default_value=0)
  batch_size = 32


  iterator = create_dataset_iterator(train_ctx, train_utterance, train_label, vocab_table, batch_size)

  sess = tf.Session()
  sess.run(tf.tables_initializer())
  sess.run(iterator.initializer)

  ctx, utterance, label = sess.run(iterator.get_next())
  print ('ctx: %s type: %s\n'%(ctx, type(ctx)))
  print ('utt: %s\n'%utterance)
  print ('Label: %s\n' % label)


if __name__ == '__main__':
  test()
