import logging, argparse, subprocess, os

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cdir', help='Candidate dir')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()