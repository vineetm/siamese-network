import logging, argparse, subprocess, os


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sdir')
    parser.add_argument('-out_txt1', default='valid.txt1')
    parser.add_argument('-out_txt2', default='valid.candidates')
    parser.add_argument('-out_scores', default='valid.scores')

    parser.add_argument('-num_jobs', default=32, type=int)
    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    logging.info(args)

    for k in range(args.num_jobs+1):
        txt1f = os.path.join(args.sdir, f'{args.out_txt1}.k{k}')
        candf = os.path.join(args.sdir, f'{args.out_txt2}.k{k}')
        scoresf = os.path.join(args.sdir, f'{args.out_scores}.k{k}')
        job_out = os.path.join(args.sdir, f'out.k{k}')

        cmd = f'jbsub -p scores -n sc-{k} -o {job_out} -q x86_1h -cores 1x1+1 -mem 4g ./eval.sh {txt1f} {candf} {scoresf}'
        status, output = subprocess.getstatusoutput(cmd)
        logging.info(output)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()