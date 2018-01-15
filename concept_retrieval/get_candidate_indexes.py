import logging, argparse
import pickle

from itertools import combinations
from build_candidate_map import build_word2cluster_map


class ConceptRetrieval(object):
    def __init__(self, cluster_map_pkl, word_clusters):
        with open(f'{cluster_map_pkl}.pkl', 'rb') as fr:
            self.cluster_map = pickle.load(fr)

        self.word2cluster, clusters = build_word2cluster_map(word_clusters)
        del clusters

        self.saved_key_map = {}
        self.new_computations = 0
        self.reused_computations = 0

    def get_cluster_index(self, word):
        return self.word2cluster[word]

    def get_covered_candidates(self, key):
        if len(key) < 1:
            return set()
        if key not in self.saved_key_map:
            candidates = self.cluster_map[key[0]]
            for cluster in key[1:]:
                candidates = candidates.intersection(self.cluster_map[cluster])
            self.saved_key_map[key] = candidates
            self.new_computations += 1
        else:
            self.reused_computations += 1

        #logging.info(f'Key: {key} #Cs: {len(self.saved_key_map[key])}')
        return self.saved_key_map[key]

    def get_best_candidates(self, clusters, max_candidates, candidates_set=None):
        candidates = []
        if not candidates_set:
            candidates_set = set()
        for drop in range(len(clusters)):
            for key in combinations(clusters, len(clusters) - drop):
                for candidate in self.get_covered_candidates(key):
                    if candidate in candidates_set:
                        continue
                    candidates.append(candidate)
                    candidates_set.add(candidate)
                    if len(candidates) == max_candidates:
                        return candidates, candidates_set
        return candidates, candidates_set

    def usage_stats(self):
        return f'Reused/New: {self.reused_computations}/{self.new_computations}'


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-word_clusters')
    parser.add_argument('-cluster_map')

    parser.add_argument('-preds')
    parser.add_argument('-preds_candidates_pkl')

    parser.add_argument('-max_candidates', default=100, type=int)
    parser.add_argument('-reverse', action='store_true', default=False)

    parser.add_argument('-num_translations', default=5, type=int)
    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    logging.info(args)

    cr = ConceptRetrieval(args.cluster_map, args.word_clusters)

    all_candidates = []
    candidates = []
    candidates_set = None
    for index, pred in enumerate(open(args.preds)):
        if index % args.num_translations == 0:
            if candidates:
                all_candidates.append(candidates)
                logging.info(f'I: {index} cl: {clusters} #Cs: {len(candidates)} Usage: {cr.usage_stats()}')
                candidates = []
                candidates_set = None

        #Get clusters in sorted order
        clusters = sorted([cr.word2cluster[w] for w in set(pred.strip().split())], reverse=args.reverse)
        new_candidates, candidates_set = cr.get_best_candidates(clusters, args.max_candidates, candidates_set)
        candidates.extend(new_candidates)

    assert candidates
    all_candidates.append(candidates)
    logging.info(f'I: {index} cl: {clusters} #Cs: {len(candidates)} Usage: {cr.usage_stats()}')

    with open(f'{args.preds_candidates_pkl}.pkl', 'wb') as fw:
        pickle.dump(all_candidates, fw)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()