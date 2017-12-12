import logging, argparse

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-preds')
  parser.add_argument('-gt')
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  perfect_match = 0
  total_s = 0

  c_tp = 0.0
  c_fp = 0.0
  c_fn = 0.0

  for preds, gt in zip(open(args.preds), open(args.gt)):
    preds = set(preds.split())
    gt = set(gt.split())

    tp = gt.intersection(preds)
    fp = preds - gt
    fn = gt - preds

    if len(fp) == 0 and len(fn) == 0:
      perfect_match += 1

    c_tp += len(tp)
    c_fp += len(fp)
    c_fn += len(fn)

    total_s += 1

  pr = c_tp / (c_tp + c_fp)
  re = c_tp / (c_tp + c_fn)
  f1 = (2 * pr * re) / (pr + re)

  logging.info('F1: %.4f Pr: %.4f Re: %.4f TP: %d FP: %d FN: %d Perfect:%d/%d'%
               (f1, pr, re, int(c_tp), int(c_fp), int(c_fn), perfect_match, total_s))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()