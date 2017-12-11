import argparse, logging, numpy as np

from commons import SW_BIN
from commons import create_cluster_map, get_cluster_words

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-word_clusters')
  parser.add_argument('-candidates')
  parser.add_argument('-candidates_bin', help='Bin assignment for candidate')
  parser.add_argument('-bin_members', help='Candidates for each bin')

  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  word2cluster = create_cluster_map(args.word_clusters)
  logging.info('#Words which have clusters: %d'%(len(word2cluster)))

  fw_candidate_bin = open(args.candidates_bin, 'w')

  bins = {}
  index = 0
  for candidate in open(args.candidates):
    cluster_words = get_cluster_words(candidate, word2cluster)
    if len(cluster_words) == 0:
      bin = SW_BIN
    else:
      #We want unique clusters
      sorted_clids = np.sort(list(set([word2cluster[word] for word in cluster_words])))
      bin = '%s'%' '.join([str(clid) for clid in sorted_clids])

    if bin not in bins:
      bins[bin] = []
    bins[bin].append(index)
    fw_candidate_bin.write('%s\n'%bin)

    index += 1
    if index % 10000 == 0:
      logging.info('Candidate: %d Bins: %d'%(index, len(bins)))

  with open(args.bin_members, 'w') as fw:
    for bin in bins:
      fw.write('%s;%s\n'%(bin, ' '.join([str(candidate_index) for candidate_index in bins[bin]])))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()