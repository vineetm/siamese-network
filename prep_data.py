import tensorflow as tf
import argparse, os, csv
import numpy as np
from collections import Counter

logging = tf.logging
logging.set_verbosity(logging.INFO)


CSV = 'csv'
UNK = 'UNK'

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('raw_data_dir', help='Raw Ubuntu data directory')
  parser.add_argument('out_dir', help='Data directory')
  parser.add_argument('-seed', default=1543, type=int)

  parser.add_argument('-train', default='train')
  parser.add_argument('-valid', default='valid')
  parser.add_argument('-test',  default='test')

  #Used for training
  parser.add_argument('-txt1', default='txt1')
  parser.add_argument('-txt2', default='txt2')
  parser.add_argument('-labels', default='labels')

  parser.add_argument('-vocab', default='vocab.txt')
  parser.add_argument('-only_valid', action='store_true')

  args = parser.parse_args()
  return args


def process_train_data(data_dir, out_dir, train_suffix, text1, text2, labels):
  train_src_file = os.path.join(data_dir, '%s.%s'%(train_suffix, CSV))

  logging.info('Reading train file: %s'%train_src_file)
  fw_txt1 = open(os.path.join(out_dir, '%s.%s' % (train_suffix, text1)), 'w')
  fw_txt2 = open(os.path.join(out_dir, '%s.%s' % (train_suffix, text2)), 'w')
  fw_labels = open(os.path.join(out_dir, '%s.%s' % (train_suffix, labels)), 'w')

  with open(train_src_file) as fr:
  #Get CSV Reader, and skip header
    reader = csv.reader(fr)
    next(reader)

    for row in reader:
      assert len(row) == 3
      text1, text2, label = row

      text1 = text1.lower()
      text2 = text2.lower()

      #Text1 is a combination of previous utterances
      fw_txt1.write('%s\n'%(' '.join(text1.split(','))))

      fw_txt2.write('%s\n'%text2)
      fw_labels.write('%s\n'%label)

  fw_txt1.close()
  fw_txt2.close()
  fw_labels.close()


def process_valid_data(data_dir, out_dir, valid_suffix, text1, text2, labels):
  valid_src_file = os.path.join(data_dir, '%s.%s' % (valid_suffix, CSV))
  logging.info('Reading valid file: %s' % valid_src_file)

  with open(valid_src_file) as fr:
    reader = csv.reader(fr)
    row = next(reader)
    num_text2 = len(row) - 1
    logging.info('num_text2: %d'%num_text2)

    fw_txt1 = open(os.path.join(out_dir, '%s.%s' % (valid_suffix, text1)), 'w')
    fw_txt2 = open(os.path.join(out_dir, '%s.%s' % (valid_suffix, text2)), 'w')
    fw_labels = open(os.path.join(out_dir, '%s.%s' % (valid_suffix, labels)), 'w')

    lastLabel = 0
    for row in reader:
      fw_txt1.write('%s\n'%row[0].lower())
      if lastLabel == 0:
        fw_txt2.write('%s\n' % row[1].lower())
        lastLabel = 1
      else:
        neg_index = np.random.randint(1, 10)
        fw_txt2.write('%s\n' % row[neg_index].lower())
        lastLabel = 0
      fw_labels.write('%d\n' % lastLabel)

'''
Text1: context
Text2: split into 10 files .p0, .p1, .p2, ... 
'''
def process_repeat_valid_data_for_retrieval(data_dir, out_dir, valid_suffix, text1, text2):
  valid_src_file = os.path.join(data_dir, '%s.%s' % (valid_suffix, CSV))
  logging.info('Reading valid file: %s'%valid_src_file)

  with open(valid_src_file) as fr:
    reader = csv.reader(fr)
    row = next(reader)
    num_text2 = len(row) - 1
    logging.info('num_text2: %d'%num_text2)

    fw_txt1 = open(os.path.join(out_dir, 'rp%s.%s' % (valid_suffix, text1)), 'w')
    fw_txt2 = open(os.path.join(out_dir, 'rp%s.%s' % (valid_suffix, text2)), 'w')

    for row in reader:
      assert len(row) == num_text2 + 1

      for pnum, part in enumerate(row[1:]):
        part = part.lower()
        fw_txt1.write('%s\n' % (' '.join(row[0].lower().split(','))))
        fw_txt2.write('%s\n'%part)

'''
Text1: context
Text2: split into 10 files .p0, .p1, .p2, ... 
'''
def process_valid_data_for_retrieval(data_dir, out_dir, valid_suffix, text1, text2):
  valid_src_file = os.path.join(data_dir, '%s.%s' % (valid_suffix, CSV))
  logging.info('Reading valid file: %s'%valid_src_file)

  with open(valid_src_file) as fr:
    reader = csv.reader(fr)
    row = next(reader)
    num_text2 = len(row) - 1
    logging.info('num_text2: %d'%num_text2)

    fw_txt1 = open(os.path.join(out_dir, 'r%s.%s' % (valid_suffix, text1)), 'w')
    fw_txt2 = []
    for pnum in range(num_text2):
      fw = open(os.path.join(out_dir, 'r%s.%s.p%d' % (valid_suffix, text2, pnum)), 'w')
      fw_txt2.append(fw)

    for row in reader:
      assert len(row) == num_text2 + 1
      fw_txt1.write('%s\n'%(' '.join(row[0].lower().split(','))))

      for pnum, part in enumerate(row[1:]):
        part = part.lower()
        fw_txt2[pnum].write('%s\n'%part)


def create_vocab(out_dir, train_suffix, text1, text2, vocab):
  fr_txt1 = open(os.path.join(out_dir, '%s.%s' % (train_suffix, text1)))
  fr_txt2 = open(os.path.join(out_dir, '%s.%s' % (train_suffix, text2)))

  words_buffer = []
  counter = Counter(words_buffer)

  index = 0
  for line1, line2 in zip(fr_txt1, fr_txt2):
    words_buffer.extend(line1.split())
    words_buffer.extend(line2.split())

    index += 1
    if len(words_buffer) > 1000000:
      counter.update(words_buffer)
      words_buffer = []
      logging.info('I: %d Vocab: %d'%(index, len(counter)))

  if len(words_buffer) > 0:
    counter.update(words_buffer)
  logging.info('Final Vocab: %d' % len(counter))

  fw_vocab = open(os.path.join(out_dir, vocab), 'w')
  fw_vocab.write('%s\n'%UNK)
  for w, _ in counter.most_common():
    fw_vocab.write('%s\n'%w)


def main():
  args = setup_args()
  np.random.seed(args.seed)
  logging.info(args)

  # logging.info('Creating Train')
  # process_train_data(args.raw_data_dir, args.out_dir, args.train, args.txt1, args.txt2, args.labels)
  #
  # logging.info('Creating Valid')
  # process_valid_data(args.raw_data_dir, args.out_dir, args.valid, args.txt1, args.txt2, args.labels)
  #
  # logging.info('Creating Valid for retrieval')
  # process_valid_data_for_retrieval(args.raw_data_dir, args.out_dir, args.valid, args.txt1, args.txt2)
  #
  # logging.info('Creating Vocab from training data')
  # create_vocab(args.out_dir, args.train, args.txt1, args.txt2, args.vocab)

  process_repeat_valid_data_for_retrieval(args.raw_data_dir, args.out_dir, args.valid, args.txt1, args.txt2)

if __name__ == '__main__':
  main()