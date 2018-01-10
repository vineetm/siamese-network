import logging, argparse, os


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cdir')
    parser.add_argument('-numc', default=19560, type=int)
    parser.add_argument('-num_jobs', default=32, type=int)

    parser.add_argument('-sdir')
    parser.add_argument('-out_txt1', default='valid.txt1')
    parser.add_argument('-out_txt2', default='valid.candidates')
    parser.add_argument('-out_map', default='valid.map')
    args = parser.parse_args()
    return args


def get_file_writers(sdir, out_txt1, out_txt2, out_map, index):
    fw_txt1 = open(os.path.join(sdir, f'{out_txt1}.k{index}'), 'w')
    fw_txt2 = open(os.path.join(sdir, f'{out_txt2}.k{index}'), 'w')
    fw_map = open(os.path.join(sdir, f'{out_map}.k{index}'), 'w')
    return fw_txt1, fw_txt2, fw_map


def main():
    args = setup_args()
    logging.info(args)

    num_files_per_job = args.numc // args.num_jobs
    logging.info(f'Files per job: {num_files_per_job}')

    job_index = 0
    for k in range(args.numc):
        if k % num_files_per_job == 0:
            fw_txt1, fw_txt2, fw_map = get_file_writers(args.sdir, args.out_txt1, args.out_txt2, args.out_map, job_index)
            logging.info(f'Job file: {job_index}')
            job_index += 1

        txt1f = os.path.join(args.cdir, f'{k}.txt1')
        txt2f = os.path.join(args.cdir, f'{k}.txt2')

        #Write contents of k.txt1, k.txt2 to job file
        for index, (txt1, txt2) in enumerate(zip(open(txt1f), open(txt2f))):
            fw_txt1.write(txt1)
            fw_txt2.write(txt2)
            fw_map.write(f'{k},{index}\n')




if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()