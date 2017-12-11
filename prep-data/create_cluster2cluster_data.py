import logging, argparse
import numpy as np

from commons import create_cluster_map, EOU, EOT

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-word_clusters')
  parser.add_argument('-txt1')
  parser.add_argument('-txt2')
  parser.add_argument('-labels', default=None)

  parser.add_argument('-out_txt1')
  parser.add_argument('-out_txt2')
  parser.add_argument('-out_vocab', default=None)

  parser.add_argument('-uniq', default=False, action='store_true')
  args = parser.parse_args()
  return args


def get_clusters_for_turn(turn, word2cluster):
  clusters = list(set([word2cluster[word] for word in turn.split() if word in word2cluster]))
  sorted_clusters_int = np.sort(clusters)
  sorted_clusters = [str(clid) for clid in sorted_clusters_int]
  return sorted_clusters


def get_clusters_for_turns(sentence, word2cluster, uniq):
  turns = sentence.split(EOT)
  out_clusters = []
  if uniq:
    all_words = []
    for turn in turns:
      all_words.extend(turn.split())
    clusters = list(set([word2cluster[word] for word in all_words if word in word2cluster]))
    sorted_clusters_int = np.sort(clusters)
    sorted_clusters = [str(clid) for clid in sorted_clusters_int]
    return sorted_clusters

  for turn in turns:
    sorted_clusters = get_clusters_for_turn(turn, word2cluster)
    if len(sorted_clusters) > 0:
      sorted_clusters.append(EOT)
      out_clusters.extend(sorted_clusters)
  return out_clusters


def process_pair(txt1, txt2, word2cluster, fw_txt1, fw_txt2, uniq):
  txt1_clusters = get_clusters_for_turns(txt1, word2cluster, uniq)
  if len(txt1_clusters) == 0:
    return

  txt2_clusters = get_clusters_for_turn(txt2, word2cluster)
  if len(txt2_clusters) == 0:
    return

  fw_txt1.write('%s\n'%' '.join(txt1_clusters))
  fw_txt2.write('%s\n'%' '.join(txt2_clusters))


def main():
  args = setup_args()
  logging.info(args)

  word2cluster, num_clusters = create_cluster_map(args.word_clusters)
  logging.info('#Words with clusters: %d Clusters: %d'%(len(word2cluster), num_clusters))

  fw_txt1 = open(args.out_txt1, 'w')
  fw_txt2 = open(args.out_txt2, 'w')

  if args.labels is None:
    for txt1, txt2 in zip(open(args.txt1), open(args.txt2)):
      process_pair(txt1, txt2, word2cluster, fw_txt1, fw_txt2, args.uniq)
  else:
    for txt1, txt2, label in zip(open(args.txt1), open(args.txt2), open(args.labels)):
      label = int(label)
      if label == 0:
        continue
      process_pair(txt1, txt2, word2cluster, fw_txt1, fw_txt2, args.uniq)

  if args.out_vocab is None:
    return

  logging.info('Creating vocab')
  with open(args.out_vocab, 'w') as fw:
    fw.write('%s\n'%EOT)
    for clid in range(num_clusters):
      fw.write('%d\n'%clid)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()