import argparse, logging

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-preds')
  parser.add_argument('-gt')
  parser.add_argument('-num_translations', default=8, type=int)
  args = parser.parse_args()
  return args


def read_gt(gt_file):
  gt = []
  for line in open(gt_file):
    gt.append(set(line.split()))
  return gt


def main():
  args = setup_args()
  logging.info(args)

  gt = read_gt(args.gt)
  logging.info('#Data points: %d'%len(gt))

  line_num = 0
  datum_num = 0
  preds = set()

  all_tp, all_fp, all_fn = 0., 0., 0.

  for line in open(args.preds):
    preds |= set(line.split())

    line_num += 1
    if line_num == args.num_translations:
      tp = gt[datum_num].intersection(preds)
      fp = preds - gt[datum_num]
      fn = gt[datum_num] - preds

      all_tp += len(tp)
      all_fp += len(fp)
      all_fn += len(fn)

      datum_num += 1
      line_num = 0
      preds = set()

  assert len(preds) == 0

  pr = all_tp / (all_tp + all_fp)
  re = all_tp / (all_tp + all_fn)
  f1 = (2 * pr * re) / (pr + re)

  logging.info('Pr: %.4f Re: %.4f F1: %.4f'%(pr, re, f1))
  logging.info('TP: %d FP: %d FN: %d'%(int(all_tp), int(all_fp), int(all_fn)))

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()