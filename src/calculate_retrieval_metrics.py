import argparse, logging
import pickle as pkl
import numpy as np

from eval_all_candidates import convert_to_numpy_array

NC = [10, 100, 500, 1000, 5000]


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-scores')
  parser.add_argument('-out_metrics')
  args = parser.parse_args()
  return args


def get_retrieval_metrics(out_metrics, list_scores):
  scores = convert_to_numpy_array(list_scores)
  fw = open(out_metrics, 'w')

  N = scores.shape[0]
  for nc in NC:
    rank_gt = nc - np.argmin(np.argsort(scores[:,:nc], axis=1), axis=1)
    assert rank_gt.shape[0] == scores.shape[0]
    r1 = np.true_divide(sum(rank_gt == 1), N)
    r2 = np.true_divide(sum(rank_gt <= 2), N)
    r5 = np.true_divide(sum(rank_gt <= 5), N)
    mrr = np.average(1.0 / rank_gt)

    logging.info('(1 in %4d): R@1=%.4f R@2:%.4f R@5: %.4f MRR: %.4f'%(nc, r1, r2, r5, mrr))
    fw.write('(1 in %4d): R@1=%.4f R@2:%.4f R@5: %.4f MRR: %.4f\n'%(nc, r1, r2, r5, mrr))

def main():
  args = setup_args()
  logging.info(args)

  with open(args.scores, 'rb') as fr:
    list_scores = pkl.load(fr)
  logging.info('Num scores: %d'%len(list_scores))

  get_retrieval_metrics(args.out_metrics, list_scores)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()
