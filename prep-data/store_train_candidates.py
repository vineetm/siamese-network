import argparse, logging

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-all_candidates', default=None)
  parser.add_argument('-all_labels', default=None)
  parser.add_argument('-out_candidates', default=None)
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  num = 0
  candidates_set = set()
  for candidate, label in zip(open(args.all_candidates), open(args.all_labels)):
    label = int(label)

    #Skip candidate with label 0
    if label == 0:
      continue
    num += 1
    candidates_set.add(candidate.strip())
  logging.info('Unique: %d/%d'%(len(candidates_set), num))

  with open(args.out_candidates, 'w') as fw:
    for candidate in candidates_set:
      fw.write('%s\n'%candidate)
  logging.info('Wrote candidates to %s'%args.out_candidates)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()