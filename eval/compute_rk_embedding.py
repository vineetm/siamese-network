'''
Compute Ubuntu R@k metrics using average cosine similarity of word embeddings
'''
import argparse, logging, subprocess
from gensim.models import KeyedVectors


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-word2vec',
                        default='/u/vineeku6/data/word-vectors/word2vec/GoogleNews-vectors-negative300.bin')
    parser.add_argument('-predictions')
    parser.add_argument('-candidates')
    parser.add_argument('-num', default=10, type=int, help='#Candidates per example')
    return parser.parse_args()


def main():
    args = setup_args()
    logging.info(args)

    # Find number of predictions
    cmd = f'wc -l < {args.predictions}'
    num_predictions = subprocess.getoutput(cmd)
    logging.info(f'Predictions: {num_predictions}')

    #Load Word2vec
    w2v = KeyedVectors.load_word2vec_format(args.word2vec, binary=True)
    logging.info(f'Word vectors found for {len(w2v.vocab)}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()