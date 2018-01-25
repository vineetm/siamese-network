import logging, argparse, os, re

NUM_CANDIDATES = [1000, 500, 300, 200, 100, 50, 20, 10, 5, 2]
WEIGHTS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-metrics_dir', help='Metrics directory')
    return parser.parse_args()


def main():
    args = setup_args()
    logging.info(args)

    max_ef1, max_af1, max_avg = 0., 0., 0.

    for k in NUM_CANDIDATES:
        sub_dir = os.path.join(args.metrics_dir, f'k{k}')

        for wt in WEIGHTS:
            metric_file = os.path.join(sub_dir, f'c.{wt}.out')

            with open(metric_file) as fr:
                lines = fr.readlines()

            lines = ''.join(lines)
            match = re.search(r'Entity F1: (0\.(\d+))', ''.join(lines))
            if match:
                ef1 = float(match.group(1))

            match = re.search(r'Activity F1: (0\.(\d+))', ''.join(lines))
            if match:
                af1 = float(match.group(1))

            avg_f1 = (af1 + ef1) / 2.0
            print(f'k:{k} wt: {wt} ef1:{ef1} af1: {af1}, avg:{avg_f1}')

            if avg_f1 > max_avg:
                max_avg = avg_f1
                max_ef1 = ef1
                max_af1 = af1
                max_k = k
                max_wt = wt
    print(f'k: {max_k} wt: {max_wt} max_avg: {max_avg} max_ef1: {max_ef1} max_af1: {max_af1}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()