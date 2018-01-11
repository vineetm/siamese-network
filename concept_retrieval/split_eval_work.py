import logging, argparse, os
import pickle

def setup_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-candidates')

    parser.add_argument('-txt1')
    parser.add_argument('-preds_candidates_pkl')

    parser.add_argument('-sdir')
    parser.add_argument('-out_txt1', default='valid.txt1')
    parser.add_argument('-out_txt2', default='valid.candidates')
    parser.add_argument('-out_map', default='valid.map')

    parser.add_argument('-num_jobs', default=128, type=int)
    parser.add_argument('-max_candidates', default=5000, type=int)
    args = parser.parse_args()
    return args


def get_file_writers(sdir, out_txt1, out_txt2, out_map, index):
    fw_txt1 = open(os.path.join(sdir, f'{out_txt1}.k{index}'), 'w')
    fw_txt2 = open(os.path.join(sdir, f'{out_txt2}.k{index}'), 'w')
    fw_map = open(os.path.join(sdir, f'{out_map}.k{index}'), 'w')
    return fw_txt1, fw_txt2, fw_map


def load_candidates(candidates_f):
    with open(candidates_f) as fr:
        return fr.readlines()


def main():
    args = setup_args()

    all_candidates = load_candidates(args.candidates)
    logging.info(f'Total Candidates: {len(all_candidates)}')

    with open(f'{args.preds_candidates_pkl}.pkl', 'rb') as fr:
        all_candidate_indexes = pickle.load(fr)

    logging.info('All_can')

    total_work = len(all_candidate_indexes) * args.max_candidates
    work_per_job = total_work // (args.num_jobs + 1)
    logging.info(f'Numc: {len(all_candidate_indexes)} Work_per_job: {work_per_job}')

    work_index = 0
    job_index = 0

    for txt1_index, txt1 in enumerate(open(args.txt1)):
        for ci, cnum in enumerate(all_candidate_indexes[txt1_index]):
            if work_index % work_per_job == 0:
                fw_txt1, fw_txt2, fw_map = get_file_writers(args.sdir, args.out_txt1, args.out_txt2, args.out_map,
                                                            job_index)
                job_index += 1
                work_index = 0
            work_index += 1
            fw_txt1.write(txt1)
            fw_txt2.write(all_candidates[int(cnum)])
            fw_map.write(f'{txt1_index},{ci}\n')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()