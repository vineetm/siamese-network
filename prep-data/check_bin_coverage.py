import argparse, logging

from commons import create_cluster_map, find_bin_key


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-word_clusters')
  parser.add_argument('-bin')
  parser.add_argument('-sentences')
  args = parser.parse_args()
  return args


def create_bin_index(bin_file):
  bin_index = {}
  index  = 0
  for bin in open(bin_file):
    bin = bin.strip()
    bin_index[bin] = index
    index += 1
  return bin_index


def main():
  args = setup_args()
  logging.info(args)

  word2cluster = create_cluster_map(args.word_clusters)
  logging.info('#Words with cluster assignments: %d'%len(word2cluster))

  bin_index = create_bin_index(args.bin)
  logging.info('#Bins: %d'%len(bin_index))

  num_found = 0
  total = 0
  num_missing = 0

  for sentence in open(args.sentences):
    bin_key = find_bin_key(sentence, word2cluster)
    if bin_key in bin_index:
      num_found += 1
    else:
      num_missing += 1

    total += 1

  logging.info('Found: %d Missing: %d Total: %d'%(num_found, num_missing, total))




if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()