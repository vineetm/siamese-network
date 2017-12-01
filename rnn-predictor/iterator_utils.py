from collections import namedtuple
import tensorflow as tf

class TrainDataIterator(namedtuple('TrainDataIterator', 'init sentence len_sentence labels weights')):
  pass

def create_train_dataset_iterator(sentences_file, vocab_sentences, labels_file, vocab_labels, max_labels, batch_size,
                                  scaling_factor):
  #Get sequence of strings
  sentences_dataset = tf.data.TextLineDataset(sentences_file)

  #Split each string into constituent words
  sentences_dataset = sentences_dataset.map(lambda sentence: tf.string_split([sentence]).values)

  #Get word index for each word
  sentences_dataset = sentences_dataset.map(lambda words: vocab_sentences.lookup(words))

  #Get all *on* labels string
  labels_dataset = tf.data.TextLineDataset(labels_file)

  #Get all *on* labels
  labels_dataset = labels_dataset.map(lambda sentence: tf.string_split([sentence]).values)

  #Get class number for each *on* label
  labels_dataset = labels_dataset.map(lambda words: vocab_labels.lookup(words))

  #Get a multi-hot representation
  labels_dataset = labels_dataset.map(
    lambda indexes: tf.reduce_max(tf.one_hot(indices=indexes, depth=max_labels), axis=0))

  #Combine the sentences and labels dataset
  dataset = tf.data.Dataset.zip((sentences_dataset, labels_dataset))

  #Sentence is variable length, get its size
  dataset = dataset.map(lambda sentence, labels: (sentence, tf.size(sentence), labels, ((labels * scaling_factor)+1)))

  #Batching
  dataset = dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]),
                                                            tf.TensorShape([None]), tf.TensorShape([None])))

  iterator = dataset.make_initializable_iterator()
  sentence, len_sentence, labels, weights = iterator.get_next()
  return TrainDataIterator(iterator.initializer, sentence, len_sentence, labels, weights)


class InferDataIterator(namedtuple('InferDataIterator', 'init sentence len_sentence')):
  pass


def create_infer_dataset_iterator(sentences_file, vocab_sentences, batch_size):
  #Get sequence of strings
  sentences_dataset = tf.data.TextLineDataset(sentences_file)

  #Split each string into constituent words
  sentences_dataset = sentences_dataset.map(lambda sentence: tf.string_split([sentence]).values)

  #Get word index for each word
  sentences_dataset = sentences_dataset.map(lambda words: vocab_sentences.lookup(words))

  #Sentence is variable length, get its size
  sentences_dataset = sentences_dataset.map(lambda sentence: (sentence, tf.size(sentence)))

  #Batching
  sentences_dataset = sentences_dataset.padded_batch(batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))

  iterator = sentences_dataset.make_initializable_iterator()
  sentence, len_sentence = iterator.get_next()
  return InferDataIterator(iterator.initializer, sentence, len_sentence)