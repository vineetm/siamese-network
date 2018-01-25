import logging, argparse, subprocess, os


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cdir', help='Candidates directory')
    parser.add_argument('-numc', help='Num candidates', default=19560, type=int)
    parser.add_argument('-q', default='x86_1h')
    parser.add_argument('-p', default='ss')
    parser.add_argument('-num_files', default=500, type=int)

    parser.add_argument('-log_dir', default='logs')
    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    logging.info(args)

    if os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    for job_num, k in enumerate(range(0, args.numc, args.num_files)):
        end = k + args.num_files-1
        if end > args.numc:
            end = args.numc-1
        logging.info(f'{job_num} {k} {end}')

        cmd = f'./eval_range.sh {args.cdir} {k} {end}'
        jbsub_cmd = f'jbsub -q {args.q} -p {args.p} -n ss.{job_num} -cores 1x1+1 -mem 6g -o {args.log_dir}/ss.{job_num}.out {cmd}'

        subprocess.getoutput(jbsub_cmd)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()