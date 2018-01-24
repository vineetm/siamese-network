'''
Compute Ubuntu R@k metrics using average cosine similarity of word embeddings
'''
import argparse, logging, subprocess
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import numpy as np


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-word2vec',
                        default='/u/vineeku6/data/word-vectors/word2vec/GoogleNews-vectors-negative300.bin')
    parser.add_argument('-candidates', default='/u/vineeku6/data/ubuntu/data/test.txt2')
    parser.add_argument('-num', default=10, type=int, help='#Candidates per example')
    parser.add_argument('-predictions')
    return parser.parse_args()


def get_embedding_average_score(sentence, w2v):
    #Find unique words in sentence
    words = set(sentence.strip().split())
    words_with_embeddings = [word for word in words if word in w2v]
    if words_with_embeddings:
        return np.average(w2v[words_with_embeddings], axis=0)
    else:
        return None


def get_candidates(fr, num):
    candidates = [fr.readline() for _ in range(num)]
    return candidates


def find_rank(scores, gt_index=0):
    for rank, index in enumerate(np.argsort(scores), start=1):
        if index == gt_index:
            return rank


def compute_distance(vec1, vec2):
    if vec2 is not None:
        return cosine(vec1, vec2)
    return 2.0


def main():
    args = setup_args()
    logging.info(args)

    # Find number of predictions
    cmd = f'wc -l < {args.predictions}'
    num_predictions = int(subprocess.getoutput(cmd))
    logging.info(f'Predictions: {num_predictions}')

    #Load Word2vec
    w2v = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
    logging.info(f'Word vectors found for {len(w2v.vocab)}')

    fr_candidates = open(args.candidates)

    num_r1, num_r2, num_r5 = 0., 0., 0.
    num_no_embedding = 0
    mrr = []
    for index, prediction in enumerate(open(args.predictions)):
        prediction_vector = get_embedding_average_score(prediction, w2v)
        candidates = get_candidates(fr_candidates, args.num)

        if prediction_vector is None:
            num_no_embedding += 1
            mrr.append(0.0)
            continue

        #Now find vectors for 10 candidates
        candidate_vectors = [get_embedding_average_score(candidate, w2v)
                            for candidate in candidates]

        candidate_scores = [compute_distance(prediction_vector, candidate_vec) for candidate_vec in candidate_vectors]
        assert len(candidate_scores) == args.num

        rank_0 = find_rank(candidate_scores)
        logging.info(f'{index}: R0: {rank_0}')

        if rank_0 == 1:
            num_r1 += 1.0
            num_r2 += 1.0
            num_r5 += 1.0
        elif rank_0 == 2:
            num_r2 += 1.0
            num_r5 += 1.0
        elif rank_0 <= 5:
            num_r5 += 1.0

        mrr.append(1.0 / rank_0)
        if index % 10 == 0:
            logging.info(f'NR1: {num_r1} NR2: {num_r2} NR5: {num_r5}  MRR: {np.average(mrr)}/ {index} '
                         f'No embedding: {num_no_embedding}')

    logging.info(f'NR1: {num_r1} NR2: {num_r2} NR5: {num_r5} / {num_predictions}')
    logging.info(f'R1: {num_r1/num_predictions} R2: {num_r2/num_predictions} R5: {num_r5/num_predictions} '
                 f'MRR: {np.average(mrr)} NE: {num_no_embedding}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()