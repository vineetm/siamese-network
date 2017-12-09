import logging, argparse, string
from nltk.corpus import stopwords
from collections import Counter

#Combination of stopwords from https://github.com/igorbrigadir/stopwords
# * 'en/spacy_gensim.txt'
# * 'en/galago_forumstop.txt'
BASE_STOPW_FILE = 'base.stopw.txt'


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-stopw_out', default='stopw_clusters.txt')
  parser.add_argument('-candidates')
  parser.add_argument('-topk', default=50, type=int, help='#Candidate words to pick as stopwords')
  args = parser.parse_args()
  return args


def create_stopw(candidates, topk):
  stopw = set()

  for line in open(BASE_STOPW_FILE):
    stopw.add(line.strip())

  for word in stopwords.words('english'):
    stopw.add(word)

  for p in string.punctuation:
    stopw.add(p)

  candidate_words = []
  for candidate in open(candidates):
    candidate_words.extend(list(set(candidate.split())))

  counter = Counter(candidate_words)
  for w, _ in counter.most_common(topk):
    stopw.add(w)

  logging.info('# Stopwords: %d'%len(stopw))
  return stopw


def main():
  args = setup_args()
  logging.info(args)

  stopw = create_stopw(args.candidates, args.topk)
  with open(args.stopw_out, 'w') as fw:
    for w in stopw:
      fw.write('%s\n'%w)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()