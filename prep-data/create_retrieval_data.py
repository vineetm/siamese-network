import argparse, logging, os
import numpy as np
from collections import OrderedDict

NUM_DISTRACTORS_LIST = [49, 99, 199, 499, 999]

CORRECT = 1
INCORRECT = 0

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-data_dir', default='data')
  parser.add_argument('-train', default='train')
  parser.add_argument('-valid', default='valid')

  parser.add_argument('-txt1', default='txt1')
  parser.add_argument('-txt2', default='txt2')
  parser.add_argument('-labels', default='labels')

  parser.add_argument('-seed', default=1543, type=int)
  parser.add_argument('-max_distractors', default=999, type=int)
  args = parser.parse_args()
  return args

'''
Get set of unique agent utterances
'''
def read_candidates(candidates_file):
  candidates_set = set()
  for line in open(candidates_file):
    candidate = line.strip()
    candidates_set.add(candidate)
  return candidates_set


def update_candidates(candidates_set, sentences_file, labels_file):
  for line, label in zip(open(sentences_file), open(labels_file)):
    line = line.strip()
    label = int(label)
    if label == 0:
      continue

    if line in candidates_set:
      candidates_set.remove(line)


def get_file_writers(num_distractors_list, txt1, txt2, labels):
  fws_txt1 = OrderedDict()
  fws_txt2 = OrderedDict()
  fws_labels = OrderedDict()

  for num_distractors in num_distractors_list:
    fws_txt1[num_distractors] = open('%s.c%d'%(txt1, num_distractors+1), 'w')
    fws_txt2[num_distractors] = open('%s.c%d'%(txt2,num_distractors+1), 'w')
    fws_labels[num_distractors] = open('%s.c%d'%(labels, num_distractors+1), 'w')
  return fws_txt1, fws_txt2, fws_labels


def write_datum(txt1, gt_txt2, candidates, indexes, fws_txt1, fws_txt2, fws_labels):
  for num_distractors in NUM_DISTRACTORS_LIST:
    #First write GT with label=1
    fws_txt1[num_distractors].write('%s'%txt1)
    fws_txt2[num_distractors].write('%s'%gt_txt2)
    fws_labels[num_distractors].write('%d\n'%CORRECT)

    for index in indexes[:num_distractors]:
      fws_txt1[num_distractors].write('%s' % txt1)
      fws_txt2[num_distractors].write('%s\n' %candidates[index])
      fws_labels[num_distractors].write('%d\n' % INCORRECT)


def main():
  args = setup_args()
  logging.info(args)

  #Set random seed, for reproducibility
  np.random.seed(args.seed)

  #All train.txt2 are candidates
  candidates_file = os.path.join(args.data_dir, '%s.%s'%(args.train, args.txt2))
  candidates_set = read_candidates(candidates_file)
  logging.info('File: %s #Candidates: %d'%(candidates_file, len(candidates_set)))

  #Remove txt2 that occur in valid with label=1
  valid_txt2 = os.path.join(args.data_dir, '%s.%s'%(args.valid, args.txt2))
  valid_labels = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.labels))
  update_candidates(candidates_set, valid_txt2, valid_labels)
  logging.info('After update #Candidates: %d'%len(candidates_set))

  valid_txt1 = os.path.join(args.data_dir, '%s.%s' % (args.valid, args.txt1))
  list_candidates = list(candidates_set)
  num_candidates = len(list_candidates)

  #Get file writers for all num_candidates
  fws_txt1, fws_txt2, fws_labels = get_file_writers(NUM_DISTRACTORS_LIST, valid_txt1, valid_txt2, valid_labels)

  datum_num = 0
  for txt1, txt2, label in zip(open(valid_txt1), open(valid_txt2), open(valid_labels)):
    label = int(label)

    #This will be hit for 9/10 data points
    if label == 0:
      continue

    indexes = np.arange(num_candidates)
    np.random.shuffle(indexes)

    #We are ony interested in max_distractors(say 999)
    indexes = indexes[:args.max_distractors]

    #Write distractors for all data points
    write_datum(txt1, txt2, list_candidates, indexes, fws_txt1, fws_txt2, fws_labels)

    datum_num += 1
    if datum_num % 10 == 0:
      logging.info('Processed datum: %d'%datum_num)


if __name__ == '__main__':

  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()