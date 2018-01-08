import argparse, logging


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-input_txt1')
  parser.add_argument('-input_txt2')
  parser.add_argument('-word_clusters')

  parser.add_argument('-output_txt1')
  parser.add_argument('-output_txt2')

  args = parser.parse_args()
  return args


def build_word2cluster_map(word_clusters):
  word2cluster = {}
  for cluster in open(word_clusters):
    members = cluster.strip().split()
    for member in members:
      if member in word2cluster:
        continue
      word2cluster[member] = members[0]
  return word2cluster


def find_clusters_in_sentence(sentence, word2cluster):
  uniq_clusters = set([word2cluster[word] for word in sentence.strip().split() if word in word2cluster])
  return uniq_clusters


def main():
  args = setup_args()
  logging.info(args)

  word2cluster = build_word2cluster_map(args.word_clusters)
  logging.info(f'Num words: {len(word2cluster)}')

  fw_txt1 = open(args.output_txt1, 'w')
  fw_txt2 = open(args.output_txt2, 'w')

  for index, (txt1, txt2) in enumerate(zip(open(args.input_txt1), open(args.input_txt2))):
    clusters_txt2 = find_clusters_in_sentence(txt2, word2cluster)
    if len(clusters_txt2) == 0:
      continue

    for cluster in clusters_txt2:
      fw_txt1.write(f'{txt1.strip()} {cluster}\n')
      fw_txt2.write(txt2)

    if index % 100000 == 0:
      logging.info(f'Done: {index}')


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()