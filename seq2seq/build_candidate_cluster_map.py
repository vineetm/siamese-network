import logging, argparse
from prep_data import build_cluster_map
from collections import defaultdict


def convert_intlist_to_string(int_array):
  return ' '.join([str(a) for a in int_array])


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-word_clusters', help='Word clusters')
  parser.add_argument('-candidates', help='candidates')
  parser.add_argument('-cluster_candidates')

  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  word2cluster, clusters =  build_cluster_map(args.word_clusters)
  logging.info('#Clusters: %d Word assigned:%d' % (len(clusters), len(word2cluster)))

  #Map from word to a list
  cluster2candidates = defaultdict(list)
  index = 0
  for candidate in open(args.candidates):
    clusters_present = set([clusters[word2cluster[word]][0] for word in candidate.split() if word in word2cluster])

    for cluster in clusters_present:
      cluster2candidates[cluster].append(index)

    index += 1

  fw = open(args.cluster_candidates, 'w')
  for index in range(len(clusters)):
    key = clusters[index][0]
    if key not in cluster2candidates:
      fw.write('\n')
    else:
      fw.write('%s\n'%convert_intlist_to_string(cluster2candidates[key]))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()