import gensim
import logging, argparse
from gensim.models import Word2Vec

#Iterator that returns words from parallel files files[0] and file[1]
#Skips file[2] if label is 0
class SentencesIterator:
  def __init__(self, files):
    assert len(files) == 3
    self.files = files

  def __iter__(self):
    for line1, line2, label in zip(open(self.files[0]), open(self.files[1]), open(self.files[2])):
      label = int(label)
      if label == 0:
        continue
      yield line1.split()
      yield line2.split()


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-files', help='comma separated list of files')
  parser.add_argument('-model', help='Save model path')
  parser.add_argument('-d', type=int)
  parser.add_argument('-workers', default=8, type=int)
  args = parser.parse_args()
  return args


def main():
  args = setup_args()
  logging.info(args)

  iterator = SentencesIterator(args.files.split(','))

  model = Word2Vec(iterator, min_count=10, size=args.d, workers=args.workers)
  model.save(args.model)



if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()