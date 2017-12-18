import argparse, logging


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-stopw', help='stopwords file')

  args = parser.parse_args()
  return args

def read_stopwords(file_name):
  stop_words = set()
  with open(file_name) as fr:
    for line in fr:
      stop_words.add(line.strip())
  return stop_words


def main():
  args = setup_args()
  logging.info(args)

  stopw = read_stopwords(args.stopw)
  logging.info('#Stopwords: %d'%len(stopw))



if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()