import argparse, logging
from collections import Counter


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-input')
  parser.add_argument('-vocab')
  parser.add_argument('-all_vocab')
  parser.add_argument('-min_count', default=50, type=int)
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  ctr = Counter()
  all_words = []
  for line in open(args.input):
    all_words.extend(line.split())

    if len(all_words) > 1000000:
      ctr.update(all_words)
      logging.info('#Vocab: %d'%len(ctr))
      all_words = []

  #Final updating counter
  if len(all_words) > 1000000:
    ctr.update(all_words)
  logging.info('Final #Vocab: %d' % len(ctr))

  fw_vocab = open(args.vocab, 'w')
  fw_all_vocab = open(args.all_vocab, 'w')

  for w, f in ctr.most_common():
    if f >= args.min_count:
      fw_vocab.write('%s\n'%w)
    fw_all_vocab.write('%s %d\n'%(w, f))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()