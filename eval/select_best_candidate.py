import argparse, logging
import numpy as np


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-candidates')
    parser.add_argument('-scores')
    parser.add_argument('-best_candidate')
    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    logging.info(args)

    candidates = []
    scores = []

    for candidate, score in zip(open(args.candidates), open(args.scores)):
        candidates.append(candidate)
        scores.append(float(score.strip()))

    logging.info(f'Num candidates: {len(candidates)}')
    max_score_index = np.argmax(scores)
    with open(args.best_candidate, 'w') as fw:
        fw.write(candidates[max_score_index])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()