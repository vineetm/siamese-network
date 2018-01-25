'''
Combine Scores from IR and Siamese
'''
import logging, argparse, os
import numpy as np

IR_SCORES = 'scores'
SIAMESE_SCORES = 'siamese_scores'


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cdir')
    parser.add_argument('-wt', default=1.0, type=float, help='fraction of Siamese score')
    parser.add_argument('-output')

    parser.add_argument('-num', default=500, type=int)
    parser.add_argument('-maxc', default=1000, type=int)
    parser.add_argument('-default', default='thanks __eou__')
    return parser.parse_args()


def normalize_data(np_array):
    max = np.max(np_array)
    min = np.min(np_array)
    #Well all datums are same!
    if max == min:
        return np.ones_like(np_array)

    return (np_array - min) / (max - min)


def main():
    args = setup_args()
    logging.info(args)

    fw = open(args.output, 'w')
    for k in range(args.num):
        txt2f = os.path.join(args.cdir, f'{k}.txt2')
        ir_scoresf = os.path.join(args.cdir, f'{k}.{IR_SCORES}')
        siamese_scoresf = os.path.join(args.cdir, f'{k}.{SIAMESE_SCORES}')

        ir_scores = []
        siamese_scores = []
        candidates = []

        for index, (txt2, ir_score, siamese_score) in enumerate(zip(open(txt2f), open(ir_scoresf), open(siamese_scoresf))):
            candidates.append(txt2.strip())
            ir_scores.append(float(ir_score))
            siamese_scores.append(float(siamese_score))

        if candidates:
            candidates = candidates[:args.maxc]
            ir_scores = ir_scores[:args.maxc]
            siamese_scores = siamese_scores[:args.maxc]

            ir_scores = np.array(ir_scores)
            ir_scores = normalize_data(ir_scores)

            siamese_scores = np.array(siamese_scores)
            siamese_scores = 1.0 / (1.0 + np.exp(-siamese_scores))
            siamese_scores = normalize_data(siamese_scores)

            weighted_scores = (args.wt * siamese_scores) + ((1.0 - args.wt) * ir_scores)
            best_score_index = np.argmax(weighted_scores)
            best_candidate = candidates[best_score_index]
            #logging.info(f'Input {k}: {best_score_index}')
        else:
            #logging.info(f'Input {k}: Use default')
            best_candidate = args.default

        fw.write(f'{best_candidate}\n')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()