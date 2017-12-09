import gensim
import logging, argparse
import pickle as pkl

from collections import OrderedDict
from gensim.models import Word2Vec

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-word2vec')
  parser.add_argument('-stopw')
  parser.add_argument('-merge_th', default=0.7, type=float, help='Merge clusters threshold')
  parser.add_argument('-cluster_out')
  args = parser.parse_args()
  return args


def read_stopwords(file):
  stopw = set()
  for line in open(file):
    stopw.add(line.strip())
  return stopw


def find_assigned_cluster(words, word2cluster):
  cluster_id = None
  to_merge_cluster_ids = set()
  for word in words:
    if word in word2cluster:
      if cluster_id is not None:
        if word2cluster[word] != cluster_id:
          to_merge_cluster_ids.add(word2cluster[word])
      else:
        cluster_id = word2cluster[word]
  return cluster_id, to_merge_cluster_ids


def main():
  args = setup_args()
  logging.info(args)

  stopw = read_stopwords(args.stopw)
  logging.info('#Stopwords: %d'%len(stopw))

  model = Word2Vec.load(args.word2vec)

  cluster_id = 0
  word_clusters = OrderedDict()
  word2cluster = {}

  for word_index in range(len(model.wv.vocab)):
    word = model.wv.index2word[word_index]

    #Ignore stopwords
    if word in stopw:
      continue

    #Find words most similar to word
    most_similar_words = [w for (w, s) in model.most_similar(word, topn=100) if w not in stopw and s > 0.7]

    #We did not find any word close enough to this, skip it!
    if len(most_similar_words) == 0:
      continue

    #See if any of these words have been assigned a cluster
    assigned_cluster_id, to_merge_cluster_ids = find_assigned_cluster(most_similar_words, word2cluster)

    if assigned_cluster_id is None:
      assert len(to_merge_cluster_ids) == 0
      cluster_id += 1
      assigned_cluster_id = cluster_id
    else:
      for clid in to_merge_cluster_ids:
        for w in word_clusters[clid]:
          word_clusters[assigned_cluster_id].add(w)
          word2cluster[w] = assigned_cluster_id
        del word_clusters[clid]

    #Add words to cluster
    most_similar_words.append(word)
    if assigned_cluster_id not in word_clusters:
      word_clusters[assigned_cluster_id] = set()

    for w in most_similar_words:
      word2cluster[w] = assigned_cluster_id
      word_clusters[assigned_cluster_id].add(w)

    if word_index %1000 == 0:
      logging.info('Processed word:%s[%d] #Clusters:%d'%(word, word_index, len(word_clusters)))


  with open(args.cluster_out, 'w') as fw:
    for cid in word_clusters:
      fw.write('%s\n'%' '.join(list(word_clusters[cid])))

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


  main()