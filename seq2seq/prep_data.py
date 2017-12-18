import argparse, logging


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-stopw', help='stopwords file')
  parser.add_argument('-word_clusters', help='Word clusters')

  args = parser.parse_args()
  return args


def build_cluster_map(file_name):
  with open(file_name) as fr:
    clusters = []
    word2cluster = {}

    for line in fr:
      cluster = line.split()
      for word in cluster:
        assert word not in word2cluster
        word2cluster[word] = len(clusters)
      clusters.append(cluster)
  return word2cluster, clusters


def read_stopwords(file_name):
  stop_words = set()
  with open(file_name) as fr:
    for line in fr:
      stop_words.add(line.strip())
  return stop_words


def main():
  args = setup_args()
  logging.info(args)

  stopw = read_stopwords(args.stopw)
  logging.info('#Stopwords: %d'%len(stopw))

  word2cluster, clusters = build_cluster_map(args.word_clusters)
  logging.info('#Clusters: %d Word assigned:%d'%(len(clusters), len(word2cluster)))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()