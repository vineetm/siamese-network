'''
Create retrieval data for R@k tasks
'''
import argparse, logging
import numpy as np
import pickle as pkl


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-candidates_pkl')
  parser.add_argument('-gt')
  parser.add_argument('-output_map')
  parser.add_argument('-random_seed', default=1543, type=int)
  parser.add_argument('-k', default=50000, type=int)
  args = parser.parse_args()
  return args


def get_k_random_values(k, txt2, candidates):
  txt2 = txt2.strip()
  random_values = []
  while True:
    rv = np.random.randint(0, len(candidates))
    if rv in random_values:
      continue

    if candidates[rv] == txt2:
      continue

    random_values.append(rv)
    if len(random_values) == k:
      return random_values

def load_candidates(candidates_pkl):
  with open(candidates_pkl, 'rb') as fr:
    candidates = pkl.load(fr)
  return candidates


def main():
  args = setup_args()
  logging.info(args)

  np.random.seed(args.random_seed)

  #Load candidates file
  candidates = load_candidates(args.candidates_pkl)
  logging.info('Num_Candidates %d'%len(candidates))

  fw_map = open(args.output_map, 'w')

  for gt in open(args.gt):
    rvs = get_k_random_values(args.k, gt, candidates)
    fw_map.write('%s\n'%','.join([str(rv) for rv in rvs]))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()

