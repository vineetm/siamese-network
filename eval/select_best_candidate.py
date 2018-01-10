import argparse, logging
import numpy as np
import os


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-candidates_dir')
    parser.add_argument('-scores_dir')
    parser.add_argument('-best_candidate_suffix')
    parser.add_argument('-num', default=10, type=int)
    args = parser.parse_args()
    return args


def select_best_candidate(candidates_f, scores_f):
    candidates = []
    scores = []

    for candidate, score in zip(open(candidates_f), open(scores_f)):
        candidates.append(candidate)
        scores.append(float(score.strip()))
    max_score_index = np.argmax(scores)
    del scores
    return candidates[max_score_index]


def main():
    args = setup_args()
    logging.info(args)

    for k in range(args.num+1):
        candidates_f = os.path.join(args.candidates_dir, f'{k}.txt2')
        scores_f = os.path.join(args.scores_dir, f'{k}.scores')
        best_candidate = select_best_candidate(candidates_f, scores_f)

        with open(os.path.join(args.scores_dir, f'{k}.{args.best_candidate_suffix}'), 'w') as fw:
            fw.write(best_candidate)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()