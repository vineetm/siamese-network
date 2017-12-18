import gensim
import logging, argparse
from gensim.models import Word2Vec

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-word2vec')
  parser.add_argument('-stopw')
  parser.add_argument('-merge_th', default=0.7, type=float, help='Merge clusters threshold')
  parser.add_argument('-cluster_out')
  parser.add_argument('-max_vocab', default=15000, type=int)
  args = parser.parse_args()
  return args


def read_stopwords(file):
  stopw = set()
  for line in open(file):
    stopw.add(line.strip())
  return stopw


def main():
  args = setup_args()
  logging.info(args)

  stopw = read_stopwords(args.stopw)
  logging.info('#Stopwords: %d'%len(stopw))

  model = Word2Vec.load(args.word2vec)
  word_clusters = []
  assigned_words = set()

  for word_index in range(args.max_vocab):
    word = model.wv.index2word[word_index]

    #This already belongs to a cluster, continue
    if word in assigned_words:
      continue

    #Ignore stopwords
    if word in stopw:
      continue

    #Find words most similar to word
    most_similar_words = [w for (w, s) in model.most_similar(word, topn=500)
                          if w not in stopw and w not in assigned_words and s > args.merge_th]

    #We did not find any word close enough to this, skip it!
    if len(most_similar_words) == 0:
      continue

    cluster = [word]
    cluster.extend(most_similar_words)

    word_clusters.append(cluster)
    for w in cluster:
      assigned_words.add(w)

    if word_index % 1000 == 0:
      logging.info('Processed word:%s[%d] #Clusters:%d'%(word, word_index, len(word_clusters)))


  with open(args.cluster_out, 'w') as fw:
    for word_cluster in word_clusters:
      fw.write('%s\n'%' '.join(word_cluster))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()