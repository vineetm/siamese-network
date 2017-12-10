import argparse, logging
from collections import Counter

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-candidates')
  parser.add_argument('-cluster_out')
  parser.add_argument('-candidates_map')
  parser.add_argument('-candidates_missed')
  parser.add_argument('-cluster_counts')
  args = parser.parse_args()
  return args


def build_cluster_maps(cluster_file):
  clid = 0
  word2cluster = {}
  cluster2words = {}
  for line in open(cluster_file):
    words = line.split()
    cluster2words[clid] = set(words)

    for word in words:
      word2cluster[word] = clid
    clid += 1

  return word2cluster, cluster2words


def main():
  args = setup_args()
  logging.info(args)

  word2cluster, cluster2words = build_cluster_maps(args.cluster_out)
  logging.info('#Clusters: %d'%len(cluster2words))

  cluster_map = {}
  for index in range(len(cluster2words)):
    cluster_map[index] = []

  candidate_index = 0

  fw_missed = open(args.candidates_missed, 'w')
  fw = open(args.candidates_map, 'w')

  clid_counts = []
  for candidate in open(args.candidates):
    #Find words that are assigned any cluster
    words = set(candidate.split()).intersection(word2cluster.keys())
    clids = set([word2cluster[word] for word in words])

    if len(clids) == 0:
      fw_missed.write(candidate)
      fw.write('NONE\n')
    else:
      clid_counts.extend(list(clids))
      fw.write('%s\n'%' '.join([str(i) for i in clids]))

    for clid in clids:
      cluster_map[clid].append(candidate_index)

    candidate_index += 1
    if candidate_index % 10000 == 0:
      logging.info('Processed Candidate: %d'%candidate_index)

  counter = Counter(clid_counts)
  with open(args.cluster_counts, 'w') as fw:
    for clid, count in counter.most_common():
      fw.write('%d %d\n'%(clid, count))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()