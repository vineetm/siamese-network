import argparse, logging
import numpy as np

from commons import SW_BIN, EOU, EOT
from commons import read_all_candidates


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-candidates')
  parser.add_argument('-bin_members', help='Candidates assigned to a bin, one per bin')
  parser.add_argument('-candidates_bin', help='bin assignment for a candidate, one per candidate')
  parser.add_argument('-vocab_file')
  parser.add_argument('-bin_repr', help='Representation for each bin, one per bin')
  parser.add_argument('-bin', help='Bin')

  parser.add_argument('-seed', default=1543, type=int)

  args = parser.parse_args()
  return args



def read_vocab(file_name):
  vocab = set()
  for line in open(file_name):
    vocab.add(line.strip())
  return vocab


def find_candidate_with_all_vocab(indexes, all_candidates, vocab, diff):
  selected_candidates = [all_candidates[index] for index in indexes
                         if (set(all_candidates[index].split()) - vocab) == diff]


  return selected_candidates


def find_candidates(indexes, all_candidates, vocab):
  #First try with all vocab
  selected_candidates = find_candidate_with_all_vocab(indexes, all_candidates, vocab, 0)
  if len(selected_candidates) > 0:
    return selected_candidates

  #Else, try with all vocab-1
  selected_candidates = find_candidate_with_all_vocab(indexes, all_candidates, vocab, 1)
  if len(selected_candidates) > 0:
    return selected_candidates

  # Else, try with all vocab-2
  selected_candidates = find_candidate_with_all_vocab(indexes, all_candidates, vocab, 2)
  if len(selected_candidates) > 0:
    return selected_candidates

  # Else all!
  selected_candidates = [all_candidates[index] for index in indexes]
  return selected_candidates


def main():
  args = setup_args()
  logging.info(args)

  np.random.seed(args.seed)
  candidates = read_all_candidates(args.candidates)
  logging.info('#Candidates: %d'%len(candidates))

  vocab = read_vocab(args.vocab_file)
  logging.info('Vocab: %d'%len(vocab))

  single_candidate = 0
  fw_bin_repr = open(args.bin_repr, 'w')
  fw_bin = open(args.bin, 'w')

  for line in open(args.bin_members):
    bin, indexes = line.split(';')
    indexes = indexes.strip()

    candidate_indexes = [int(index) for index in indexes.split()]
    if len(candidate_indexes) == 1:
      single_candidate += 1
      selected_index = 0
      selected_candidates = []
      selected_candidates.append(candidates[candidate_indexes[0]])
    else:
      selected_candidates = find_candidates(candidate_indexes, candidates, vocab)
      if len(selected_candidates) == 1:
        selected_index = 0
      else:
        selected_index = np.random.randint(low=0, high=len(selected_candidates))
    fw_bin_repr.write(selected_candidates[selected_index])
    fw_bin.write('%s\n'%bin)
  logging.info('Single candidate :%d '%single_candidate)

  #Now, find a representative for SW bin
  sw_indexes = []
  index = 0
  for bin_assignment in open(args.candidates_bin):
    if bin_assignment.strip() == SW_BIN:
      sw_indexes.append(index)
    index += 1

  logging.info('#SW candidates: %d'%len(sw_indexes))
  selected_candidates = find_candidates(sw_indexes, candidates, vocab)
  rn = np.random.randint(low=0, high=len(selected_candidates))
  fw_bin_repr.write(selected_candidates[rn])
  fw_bin.write('%s\n' %SW_BIN)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()