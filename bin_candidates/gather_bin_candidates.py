import argparse, logging
from collections import defaultdict
from bin_candidates import build_cluster_map, convert_intlist_to_string, OOV, STOPW

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-candidates_bin')

  parser.add_argument('-word_clusters')
  parser.add_argument('-candidates')
  parser.add_argument('-bin_clusters', help='Bin in clusters')
  parser.add_argument('-bin_words',help='Bin as words in clusters')
  parser.add_argument('-bin_members', help='Complete candidate list')
  parser.add_argument('-bin_counts', help='Complete candidate list')
  parser.add_argument('-bin_samples', help='Samples for each bin')
  parser.add_argument('-num_samples', default=5, type=int)

  args = parser.parse_args()
  return args


def get_clusters_str(bin, clusters):
  bin = bin.strip()
  if bin == STOPW or bin == OOV:
    return ''
  cluster_members = []
  for cluster in bin.split():
    cluster_members.extend(clusters[int(cluster)])
    cluster_members.append(';;')
  return ' '.join(cluster_members)


def get_samples(candidates, members, k):
  selected_candidates = []
  if k > len(members):
    k = len(members)
  for index in range(k):
    selected_candidates.append(candidates[members[index]].strip())
  return selected_candidates


def main():
  args = setup_args()
  logging.info(args)

  _, clusters = build_cluster_map(args.word_clusters)
  bin_members = defaultdict(list)
  index = 0
  for bin in open(args.candidates_bin):
    bin = bin.strip()
    bin_members[bin].append(index)
    index += 1

  bins_and_members = sorted(bin_members.items(), key=lambda k: (len(k[0].split()), len(k[1])), reverse=True)

  fw_bin_clusters = open(args.bin_clusters, 'w')
  fw_bin_words = open(args.bin_words, 'w')
  fw_bin_members = open(args.bin_members, 'w')
  fw_bin_counts = open(args.bin_counts, 'w')
  fw_bin_samples = open(args.bin_samples, 'w')

  candidates = open(args.candidates).readlines()
  for bin, members in bins_and_members:
    fw_bin_clusters.write('%s\n'%bin)
    fw_bin_words.write('%s\n'%get_clusters_str(bin, clusters))
    fw_bin_members.write('%s\n'%convert_intlist_to_string(members))
    fw_bin_counts.write('%d\n'%len(members))
    fw_bin_samples.write('%s\n'%';'.join(get_samples(candidates, members, args.num_samples)))



if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()

