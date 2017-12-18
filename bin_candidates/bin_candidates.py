from itertools import combinations
from collections import defaultdict
import numpy as np, argparse, logging

STOPW = 'stopw'
OOV = 'oov'
NA = 'na'

def read_stopwords(file_name):
  stop_words = set()
  with open(file_name) as fr:
    for line in fr:
      stop_words.add(line.strip())
  return stop_words


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


def get_clusters_for_sentence(sentence, word2cluster):
  words_with_clusters = set([word for word in sentence.split() if word in word2cluster])
  cluster_assignments = sorted(list(set([word2cluster[word] for word in words_with_clusters])))
  return cluster_assignments


def get_combinations(array, k):
    combinations_list = list(combinations(array, k))
    return combinations_list


def sort_keys_by_key_length(keys):
  return sorted(keys, key=lambda k: (len(k[0].split()), k[1]), reverse=True)


def count_assigned_candidates(index_key):
  return (np.sum(np.array(index_key) != None))


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-stopw')
  parser.add_argument('-word_clusters')
  parser.add_argument('-candidates')
  parser.add_argument('-max_partial_len', default=5, type=int)
  parser.add_argument('-min_count', default=10, type=int)
  parser.add_argument('-candidates_bin')

  args = parser.parse_args()
  return args


def is_stopw_sentence(sentence, stop_words):
    words = [word for word in sentence.split() if word not in stop_words]
    return len(words) == 0


def convert_intlist_to_string(int_array):
  return ' '.join([str(a) for a in int_array])


def build_maps(word2cluster, stop_words, candidate_file, max_partial_len, status_interval=1000, stopw_bin=STOPW, oov_bin=OOV):
    keys_candidates = defaultdict(lambda: defaultdict(set))
    index_key = []
    index_all_keys = []
    index_complete_key = []

    for candidate in open(candidate_file):
      all_keys = defaultdict(set)
      index = len(index_key)
      key = None
      clusters = get_clusters_for_sentence(candidate, word2cluster)
      if len(clusters) == 0:
        if is_stopw_sentence(candidate, stop_words):
          key = stopw_bin
        else:
          key = oov_bin
        index_complete_key.append(key)
      else:
        complete_key = convert_intlist_to_string(clusters)
        index_complete_key.append(key)
        keys_candidates[0][complete_key].add(index)
        all_keys[0].add(complete_key)

        if len(clusters) > max_partial_len+1:
          end = max_partial_len + 1
        else:
          end = len(clusters)

        for key_len in range(1, end):
          partial_keys = get_combinations(clusters, key_len)
          if len(partial_keys) == 0:
            continue
          for partial_key in partial_keys:
            partial_key_str = convert_intlist_to_string(partial_key)
            keys_candidates[key_len][partial_key_str].add(index)
            all_keys[key_len].add(partial_key_str)

      index_key.append(key)
      index_all_keys.append(all_keys)
      if len(index_key) % status_interval == 0:
        logging.info(len(index_key))

    return index_key, index_all_keys, keys_candidates, index_complete_key


def get_keys(keys_candidates, key_len, min_count):
  keys = [(key, len(keys_candidates[key_len][key])) for key in keys_candidates[key_len] if
          len(keys_candidates[key_len][key]) >= min_count]
  keys = sort_keys_by_key_length(keys)
  return keys


def assign_bin(key, keys_candidates, index_all_keys, key_len, index_key):
    num_assigned = 0
    for candidate_index in keys_candidates[key_len][key]:
      for k1 in index_all_keys[candidate_index]:
        for k2 in index_all_keys[candidate_index][k1]:
          if k1 == key_len and k2 == key:
            continue
          keys_candidates[k1][k2].remove(candidate_index)
          if len(keys_candidates[k1][k2]) == 0:
            del keys_candidates[k1][k2]
          if len(keys_candidates[k1]) == 0:
            del keys_candidates[k1]
      index_key[candidate_index] = key
      num_assigned += 1

    #Get static list of candidates that we will remove
    candidates = list(keys_candidates[key_len][key])
    for candidate_index in candidates:
      keys_candidates[key_len][key].remove(candidate_index)
      if len(keys_candidates[key_len][key]) == 0:
        del keys_candidates[key_len][key]
      if len(keys_candidates[key_len]) == 0:
        del keys_candidates[key_len]
    return num_assigned



def assign_bins(index_key, keys_candidates, index_all_keys, key_len, min_count):
    while True:
      keys = get_keys(keys_candidates, key_len, min_count)
      if len(keys) == 0:
        return count_assigned_candidates(index_key)

      num_assigned = assign_bin(keys[0][0], keys_candidates, index_all_keys, key_len, index_key)
      logging.info('Key_Len: %d Key:%s Assigned:%d Remaining: %d'% (key_len, keys[0][0], num_assigned, len(keys)-1))



def main():
  args = setup_args()
  logging.info(args)

  stop_words = read_stopwords(args.stopw)
  logging.info('Stop words: %d'%len(stop_words))

  word2cluster, clusters = build_cluster_map(args.word_clusters)
  logging.info('#Words: %d Clusters: %d'%(len(word2cluster), len(clusters)))

  logging.info('Creating Map')
  index_key, index_all_keys, keys_candidates, index_complete_key = \
    build_maps(word2cluster, stop_words, args.candidates, args.max_partial_len)
  logging.info('Created Map')

  num_covered = assign_bins(index_key, keys_candidates, index_all_keys, 0, args.min_count)
  logging.info('Num Covered complete keys: %d'%num_covered)

  for k in range(args.max_partial_len):
    key_len = args.max_partial_len - k
    num_covered = assign_bins(index_key, keys_candidates, index_all_keys, key_len, args.min_count)
    logging.info('Num Covered Key_Len: %d %d'% (key_len, num_covered))

  with open(args.candidates_bin, 'w') as fw:
    for bin in index_key:
      if bin is None:
        fw.write('%s\n'%NA)
      else:
        fw.write('%s\n'%bin)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()