import argparse, logging


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-txt1')
  parser.add_argument('-txt2')
  parser.add_argument('-labels', default=None)

  parser.add_argument('-out_txt1')
  parser.add_argument('-out_txt2')

  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  fw_txt1 = open(args.out_txt1, 'w')
  fw_txt2 = open(args.out_txt2, 'w')
  for txt1, txt2, label in zip(open(args.txt1), open(args.txt2), open(args.labels)):
    label = int(label)
    if label == 0:
      continue
    fw_txt1.write(txt1)
    fw_txt2.write(txt2)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()

