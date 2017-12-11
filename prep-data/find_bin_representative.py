import argparse, logging
import numpy as np

EOU = '__eou__'
EOT = '__eot__'


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-candidates')
  parser.add_argument('-bin_members', help='Candidates for each bin')
  parser.add_argument('-vocab_file')
  parser.add_argument('-bin_repr')
  parser.add_argument('-seed', default=1543, type=int)
  args = parser.parse_args()
  return args


def read_all_candidates(file_name):
  with open(file_name) as fr:
    candidates = fr.readlines()
  return candidates


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

  candidates = read_all_candidates(args.candidates)
  logging.info('#Candidates: %d'%len(candidates))

  vocab = read_vocab(args.vocab_file)
  logging.info('Vocab: %d'%len(vocab))

  single_candidate = 0
  not_all_vocab = 0
  fw = open(args.bin_repr, 'w')

  for line in open(args.bin_members):
    _, indexes = line.split(';')
    indexes = indexes.strip()

    candidate_indexes = [int(index) for index in indexes.split()]
    if len(candidate_indexes) == 1:
      single_candidate += 1
      fw.write(candidates[candidate_indexes[0]])
    else:
      selected_candidates = find_candidates(candidate_indexes, candidates, vocab)
      if len(selected_candidates) == 1:
        fw.write(selected_candidates[0])
      else:
        rn = np.random.randint(low=0, high=len(selected_candidates))
        fw.write(selected_candidates[rn])
  logging.info('Single:%d No_all_vocab: %d'%(single_candidate, not_all_vocab))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()