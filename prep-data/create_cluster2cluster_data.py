import logging, argparse
import numpy as np

from commons import create_cluster_map, EOU, EOT


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-word_clusters')
  parser.add_argument('-txt1')
  parser.add_argument('-txt2')
  parser.add_argument('-labels')

  parser.add_argument('-out_txt1')
  parser.add_argument('-out_txt2')
  parser.add_argument('-out_vocab')
  args = parser.parse_args()
  return args


def get_clusters_for_turn(turn, word2cluster):
  clusters = list(set([word2cluster[word] for word in turn.split() if word in word2cluster]))
  sorted_clusters_int = np.sort(clusters)
  sorted_clusters = [str(clid) for clid in sorted_clusters_int]
  return sorted_clusters


def get_clusters_for_turns(sentence, word2cluster):
  turns = sentence.split(EOT)
  out_clusters = []
  for turn in turns:
    sorted_clusters = get_clusters_for_turn(turn, word2cluster)
    if len(sorted_clusters) > 0:
      sorted_clusters.append(EOT)
      out_clusters.extend(sorted_clusters)
  return out_clusters




def main():
  args = setup_args()
  logging.info(args)

  word2cluster = create_cluster_map(args.word_clusters)
  logging.info('#Words with clusters: %d'%len(word2cluster))

  fw_txt1 = open(args.out_txt1, 'w')
  fw_txt2 = open(args.out_txt2, 'w')
  skip_no_clusters = 0

  for txt1, txt2, label in zip(open(args.txt1), open(args.txt2), open(args.labels)):
    label = int(label)
    if label == 0:
      continue

    txt1_clusters = get_clusters_for_turns(txt1, word2cluster)
    if len(txt1_clusters) == 0:
      skip_no_clusters += 1
      continue

    txt2_clusters = get_clusters_for_turn(txt2, word2cluster)
    if len(txt2_clusters) == 0:
      skip_no_clusters += 1
      continue

    fw_txt1.write('%s\n'%' '.join(txt1_clusters))
    fw_txt2.write('%s\n'%' '.join(txt2_clusters))

  logging.info('#Skipped: no clusters %d'%skip_no_clusters)




if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()