import logging, argparse


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-txt1')
  parser.add_argument('-map')
  parser.add_argument('-max_candidates', type=int, default=10)

  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  fw = open(args.map, 'w')
  line_num = 0
  for _ in open(args.txt1):
    for k in range(args.max_candidates):
      fw.write('%d,%d\n'% (line_num, k))
    line_num += 1


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()