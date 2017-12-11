#Bin for candidates that do not lie in any word cluster bin, chiefly stopwords
SW_BIN = 'SW'
EOU = '__eou__'
EOT = '__eot__'
import numpy as np

def read_all_candidates(file_name):
  with open(file_name) as fr:
    candidates = fr.readlines()
  return candidates


def create_cluster_map(cl_file):
  clid = 0
  word2cluster = {}
  for line in open(cl_file):
    words = line.split()
    for word in words:
      if word not in word2cluster:
        word2cluster[word] = clid
    clid += 1
  return word2cluster, clid


def get_cluster_words(sentence, word2cluster):
  cluster_words = set(sentence.split()).intersection(word2cluster.keys())
  return cluster_words


def find_bin_key(sentence, word2cluster):
  cluster_words = get_cluster_words(sentence, word2cluster)
  sorted_clids = np.sort(list(set([word2cluster[word] for word in cluster_words])))
  bin_key = '%s' % ' '.join([str(clid) for clid in sorted_clids])
  return bin_key