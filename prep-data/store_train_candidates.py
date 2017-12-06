'''
Store all unique training candidates.
We will use this to generate retrieval data for R@k and MRR evaluation
'''

import argparse, logging
import pickle as pkl

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-all_candidates', default=None)
  parser.add_argument('-all_labels', default=None)
  parser.add_argument('-out_candidates_txt', default=None)
  parser.add_argument('-out_candidates_pkl')

  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  candidates_set = set()
  for candidate, label in zip(open(args.all_candidates), open(args.all_labels)):
    label = int(label)

    #Skip candidate with label 0
    if label == 0:
      continue
    candidates_set.add(candidate.strip())

  logging.info('Found %d uniq candidates'%len(candidates_set))


  #So that there is correspondence between text and pkl file
  candidates_list = list(candidates_set)
  del candidates_set

  with open(args.out_candidates_txt, 'w') as fw:
    for candidate in candidates_list:
      fw.write('%s\n'%candidate)
  logging.info('Wrote candidates to %s'%args.out_candidates_txt)

  with open(args.out_candidates_pkl, 'wb') as fw:
    pkl.dump(candidates_list, fw)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()