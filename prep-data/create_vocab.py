import logging, argparse
from collections import Counter

NO_TOPIC = 'tNONE'
UNK = 'UNK'


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-txt1', help='txt1', default=None)
  parser.add_argument('-txt2', help='txt2', default=None)
  parser.add_argument('-txt3', help='txt3', default=None)
  parser.add_argument('-min_freq', default=50, type=int)
  parser.add_argument('-only_topics', default=False, action='store_true')
  parser.add_argument('-t', default=0, type=int)
  parser.add_argument('-vocab')
  parser.add_argument('-all_vocab')
  args = parser.parse_args()
  return args


def process_three_files(txt1f, txt2f, ctxf, topic_set):
  words = []
  for txt1, txt2, ctx in zip(open(txt1f), open(txt2f), open(ctxf)):
    words.extend([word for word in txt1.split() if word not in topic_set])
    words.extend([word for word in txt2.split() if word not in topic_set])
    words.extend([word for word in ctx.split() if word not in topic_set])
  counter = Counter(words)
  return counter


def process_two_files(txt1f, txt2f, topic_set):
  words = []
  for txt1, txt2 in zip(open(txt1f), open(txt2f)):
    words.extend([word for word in txt1.split() if word not in topic_set])
    words.extend([word for word in txt2.split() if word not in topic_set])
  counter = Counter(words)
  return counter


def process_single_file(txt1f, topic_set):
  words = []
  for txt1 in open(txt1f):
    words.extend([word for word in txt1.split() if word not in topic_set])
  counter = Counter(words)
  return counter


def create_topics_vocab(num_topics, only_topics, fw):
  topic_set = set()
  if num_topics == 0:
    return topic_set

  if only_topics is False:
    fw.write('%s\n' % UNK)
  fw.write('%s\n'%NO_TOPIC)

  for tnum in range(num_topics):
    topic = 't%d'%tnum
    topic_set.add(topic)
    fw.write('%s\n'%topic)

  topic_set.add(NO_TOPIC)
  return topic_set


def main():
  args = setup_args()
  logging.info(args)

  fw = open('%s' % args.vocab, 'w')
  topic_set = create_topics_vocab(args.t, args.only_topics, fw)
  # We only want topics
  if args.only_topics:
    return

  if args.txt3 is not None:
    counter = process_three_files(args.txt1, args.txt2, args.txt3, topic_set)
  elif args.txt2 is not None:
    counter = process_two_files(args.txt1, args.txt2, topic_set)
  else:
    counter = process_single_file(args.txt1, topic_set)

  all_fw_freq = open('%s.freq' % args.all_vocab, 'w')
  all_fw = open('%s' % args.all_vocab, 'w')

  #If we did not do topic write UNK
  if args.t == 0:
    fw.write('%s\n'%UNK)

  for w, freq in counter.most_common():
    all_fw.write('%s\n'%w)
    all_fw_freq.write('%s %d\n'%(w, freq))


    if freq >= args.min_freq:
      fw.write('%s\n'%w)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()