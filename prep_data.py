import csv, argparse, os
import logging
from collections import Counter
import numpy as np

np.random.seed(1543)


# Source Ubuntu corpus, cleaned up as described here
# https://github.com/brmson/dataset-sts/tree/master/data/anssel/ubuntu
# train_csv is tokenized csv file

# Fix tokenization in original data
# It seems `",` is not tokenized properly. Replace `",` -> `" ,`
UNK = 'UNK'

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('csv_dir', help='CSV directory for ubuntu dataset')
  parser.add_argument('-train_csv', default='v2-trainset.csv')
  parser.add_argument('-valid_csv', default='v2-valset.csv')
  parser.add_argument('-test_csv',  default='v2-testset.csv')

  parser.add_argument('-data_dir', default='data')
  parser.add_argument('-train', default='train')
  parser.add_argument('-all_valid', default='all.valid')
  parser.add_argument('-valid', default='valid')
  parser.add_argument('-test', default='test')
  parser.add_argument('-vocab', default='all.vocab.txt')

  parser.add_argument('-txt1', default='txt1')
  parser.add_argument('-txt2', default='txt2')
  parser.add_argument('-labels', default='labels')

  args = parser.parse_args()
  return args


def clean_txt(txt):
  txt = txt.strip().lower()
  txt = txt.replace('",', '" ,')
  return txt

def write_datum(row, fw_txt1, fw_txt2, fw_labels):
  fw_txt1.write('%s\n' % clean_txt(row[0]))
  fw_txt2.write('%s\n' % clean_txt(row[1]))
  fw_labels.write('%s\n' % clean_txt(row[2]))

def should_write(row_num, sub_sample=False, random_num=None):
  if sub_sample is False:
    return True

  index = row_num % 10
  if index == 0:
    return True

  if random_num == index:
    return True

'''
In order to keep valid distn same as train, we need to sub-sample valid data
'''
def separate_data(csv_file, data_dir, prefix, txt1_suffix, txt2_suffix, labels_suffix, sub_sample=False):
  fw_txt1 = open(os.path.join(data_dir, '%s.%s' % (prefix, txt1_suffix)), 'w')
  fw_txt2 = open(os.path.join(data_dir, '%s.%s' % (prefix, txt2_suffix)), 'w')
  fw_labels = open(os.path.join(data_dir, '%s.%s' % (prefix, labels_suffix)), 'w')

  row_num = 0
  rn = None
  with open(csv_file) as fr:
    reader = csv.reader(fr)
    for row in reader:
      assert len(row) == 3
      if sub_sample and row_num % 10 == 0:
          rn = np.random.randint(1, 10)

      if should_write(row_num, sub_sample, random_num=rn):
        write_datum(row, fw_txt1, fw_txt2, fw_labels)

      row_num += 1

  fw_txt1.close()
  fw_txt2.close()
  fw_labels.close()
  logging.info('Csv: %s Rows: %d'%(csv_file, row_num))


def build_vocab(data_dir, vocab_suffix, prefix, txt1_suffix, txt2_suffix):
  fr_txt1 = open(os.path.join(data_dir, '%s.%s' % (prefix, txt1_suffix)))
  fr_txt2 = open(os.path.join(data_dir, '%s.%s' % (prefix, txt2_suffix)))

  words_buffer = []
  ctr = Counter(words_buffer)
  for line1, line2 in zip(fr_txt1, fr_txt2):
    words_buffer.extend(line1.split())
    words_buffer.extend(line2.split())

    if len(words_buffer) > 1000000:
      ctr.update(words_buffer)
      words_buffer = []

  if len(words_buffer) > 0:
    ctr.update(words_buffer)
    del words_buffer

  fw_vocab = open(os.path.join(data_dir, vocab_suffix), 'w')
  fw_vocab.write('%s\n'%UNK)

  for w, _ in ctr.most_common():
    fw_vocab.write('%s\n'%w)

  fr_txt1.close()
  fr_txt2.close()
  fw_vocab.close()


def main():
  args = setup_args()
  logging.info(args)

  #Create data dir
  if not os.path.exists(args.data_dir):
    logging.info('Creating %s'%args.data_dir)
    os.mkdir(args.data_dir)

    # Separate out txt1, txt2 and labels for training data
  train_csv = os.path.join(args.csv_dir, args.train_csv)
  separate_data(train_csv, args.data_dir, args.train, args.txt1, args.txt2, args.labels)

  #Build vocab from train.txt1 and train.txt2
  build_vocab(args.data_dir, args.vocab, args.train, args.txt1, args.txt2)

  #Separate out txt1, txt2 and labels for validation data
  valid_csv = os.path.join(args.csv_dir, args.valid_csv)
  separate_data(valid_csv, args.data_dir, args.valid, args.txt1, args.txt2, args.labels, sub_sample=True)

  separate_data(valid_csv, args.data_dir, args.all_valid, args.txt1, args.txt2, args.labels)

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()