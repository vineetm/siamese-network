import logging, argparse

NO_TOPIC = 'tNONE'

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-gt', default=None)
  parser.add_argument('-preds', default=None)
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  tp = 0.0
  fp = 0.0
  fn = 0.0
  gt_match = 0
  total = 0

  for gt, pred in zip(open(args.gt), open(args.preds)):
    gt_topics = set(gt.strip().split())
    pred_topics = set(gt.strip().split())

    topics_tp = gt_topics.intersection(pred_topics)
    topics_fp = pred_topics - gt_topics
    topics_fn = gt_topics - pred_topics

    if len(topics_fp) > 1:
      gt_match += 1

    tp += len(topics_tp)
    fp += len(topics_fp)
    fn += len(topics_fn)

    total += 1

  pr = tp / (tp + fp)
  re = tp / (tp + fn)
  f1 = 2 * pr * re / (pr + re)
  logging.info('F1: %.4f Pr: %.4f Re: %.4f Gt: %d/%d'%(f1, pr, re, gt_match, total))

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()