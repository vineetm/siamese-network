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

    parser.add_argument('-stopw', default='/u/vineeku6/data/ubuntu/data/stopw_word2vec.txt')
    parser.add_argument('-maxc', default=-1, type=int)
    args = parser.parse_args()
    return args


def get_file_writers(sdir, out_txt1, out_txt2, out_map, index):
    fw_txt1 = open(os.path.join(sdir, f'{out_txt1}.k{index}'), 'w')
    fw_txt2 = open(os.path.join(sdir, f'{out_txt2}.k{index}'), 'w')
    fw_map = open(os.path.join(sdir, f'{out_map}.k{index}'), 'w')
    return fw_txt1, fw_txt2, fw_map


def read_stopwords(stopw_file):
    stopw = set()
    for line in open(stopw_file):
        stopw.add(line.strip())
    return stopw

def is_stopw_sentence(sentence, stopw):
    words = set(sentence.split())
    rem_words = words - stopw
    if rem_words:
        return False
    return True


def main():
    args = setup_args()
    logging.info(args)

    num_files_per_job = args.numc // args.num_jobs
    logging.info(f'Files per job: {num_files_per_job}')

    stopw = read_stopwords(args.stopw)
    logging.info(f'#Stopwords: {len(stopw)}')

    job_index = 0

    num_skipped = 0
    num_candidates = 0
    for k in range(args.numc):
        if k % num_files_per_job == 0:
            fw_txt1, fw_txt2, fw_map = get_file_writers(args.sdir, args.out_txt1, args.out_txt2, args.out_map, job_index)
            logging.info(f'Job file: {job_index}')
            job_index += 1

        txt1f = os.path.join(args.cdir, f'{k}.txt1')
        txt2f = os.path.join(args.cdir, f'{k}.txt2')

        #Write contents of k.txt1, k.txt2 to job file
        for index, (txt1, txt2) in enumerate(zip(open(txt1f), open(txt2f))):
            if is_stopw_sentence(txt2, stopw):
                num_skipped += 1
                continue

            if args.maxc > 0 and num_candidates >= args.maxc:
                continue

            fw_txt1.write(txt1)
            fw_txt2.write(txt2)
            fw_map.write(f'{k},{index}\n')
            num_candidates += 1

        logging.info(f'D: {k} {num_candidates} Skipped: {num_skipped}')
        num_skipped = 0
        num_candidates = 0


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()