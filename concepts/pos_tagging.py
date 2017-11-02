from textblob import TextBlob as tb
import argparse, os
import tensorflow as tf
from collections import Counter

logging = tf.logging
logging.set_verbosity(logging.INFO)


VALID_POS = set(['NN', 'VB'])

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_file', help='txt2 data')
  parser.add_argument('-eou', default='__eou__')

  parser.add_argument('-out_pos', default='pos')
  parser.add_argument('-out_np', default='phrases')

  parser.add_argument('-out_concepts', default='concepts')
  parser.add_argument('-out_np_concepts', default='np.concepts')
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  concepts = []
  np_concepts = []
  fw_pos = open('%s.%s'%(args.data_file, args.out_pos), 'w')
  fw_np = open('%s.%s' % (args.data_file, args.out_np), 'w')

  for line in open(args.data_file):
    sentences = line.strip().split(args.eou)[:-1]

    #Convert each sentence to a blob
    blobs = [tb(s) for s in sentences]

    pos_tags = []
    noun_phrases = []
    for blob in blobs:
      pos_tags.extend(blob.tags)
      noun_phrases.extend(blob.noun_phrases)

    np_concepts.extend(noun_phrases)
    fw_pos.write(' '.join(['%s %s'%(pos_tag[0], pos_tag[1]) for pos_tag in pos_tags]) + '\n')
    fw_np.write(';'.join(noun_phrases) + '\n')

    #Filter for nouns and verbs
    concepts.extend([pos_tag[0] for pos_tag in pos_tags if pos_tag[1][:2] in VALID_POS])

  fw_concepts = open('%s.%s' % (args.data_file, args.out_concepts), 'w')

  counter = Counter(concepts)
  for w, f in counter.most_common():
    fw_concepts.write('%s %d\n'%(w, f))
  del concepts

  fw_np_concepts = open('%s.%s' % (args.data_file, args.out_np_concepts), 'w')
  counter = Counter(np_concepts)
  for w, f in counter.most_common():
    fw_np_concepts.write('%s %d\n' % (w, f))


if __name__ == '__main__':
  main()