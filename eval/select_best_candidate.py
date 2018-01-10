import argparse, logging, os
import numpy as np

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdir')
    parser.add_argument('-out_map', default='valid.map')
    parser.add_argument('-out_txt2', default='valid.candidates')
    parser.add_argument('-out_scores', default='valid.scores')
    parser.add_argument('-num_jobs', default=32, type=int)
    parser.add_argument('-numc', default=19560, type=int)

    parser.add_argument('-topc', default='valid.topc')
    args = parser.parse_args()
    return args


def find_best_candidate(scores, candidates):
    assert len(scores) == len(candidates)
    if not candidates:
        return '', -1
    best_index = np.argmax(scores)
    return candidates[best_index], best_index


def main():
    args = setup_args()
    logging.info(args)

    with open(os.path.join(args.sdir, args.topc), 'w') as fw:
        scores = []
        candidates = []
        last_index = 0

        for job_num in range(args.num_jobs+1):
            scores_f = os.path.join(args.sdir, f'{args.out_scores}.k{job_num}')
            map_f = os.path.join(args.sdir, f'{args.out_map}.k{job_num}')
            cand_f = os.path.join(args.sdir, f'{args.out_txt2}.k{job_num}')

            for score, map, candidate in zip(open(scores_f), open(map_f), open(cand_f)):
                index, _ = [int(a) for a in map.strip().split(',')]

                if index != last_index:
                    if index != last_index+1:
                        fw.write('\n')
                    best_candidate, best_index = find_best_candidate(scores, candidates)
                    logging.info(f'Job: {job_num} {last_index}: {best_index}')
                    fw.write(f'{best_candidate.strip()}\n')
                    scores = []
                    candidates = []
                    last_index = index
                scores.append(float(score))
                candidates.append(candidate)
        if scores:
            best_candidate, best_index = find_best_candidate(scores, candidates)
            fw.write(f'{best_candidate.strip()}\n')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()