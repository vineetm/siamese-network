import argparse, logging, os, subprocess


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-candidates_dir')
    parser.add_argument('-scores_dir')
    parser.add_argument('-q', default='x86_1h')
    parser.add_argument('-mem', default='4g')
    parser.add_argument('-num_jobs', default=10, type=int)
    parser.add_argument('-jobs_file')
    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    logging.info(args)

    with open(args.jobs_file, 'w') as fw:
        fw.write(f'#!/bin/sh\n')
        for k in range(args.num_jobs +1):
            ctx_file = os.path.join(args.candidates_dir, f'{k}.txt1')
            cand_file = os.path.join(args.candidates_dir, f'{k}.txt2')

            scores_file = os.path.join(args.scores_dir, f'{k}.scores')
            out_file = os.path.join(args.scores_dir, f'{k}.out')

            cmd = f'jbsub -p scores -q {args.q} -mem {args.mem} -cores 1x1+1 -o {out_file} ./eval.sh {ctx_file} ' \
                  f'{cand_file} {scores_file}'
            fw.write(f'{cmd}\n')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()