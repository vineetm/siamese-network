'''
Replace target utterance with bin representative
'''
import argparse, logging
import numpy as np


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-bin', help='Bin Key file')
  parser.add_argument('-binr', help='Bin representative file, one per bin key')

  parser.add_argument('-candidates')
  parser.add_argument('-candidates_bin', help='Bin assignment for candidate')

  parser.add_argument('-seed', default=1543, type=int)

  parser.add_argument('-txt1')
  parser.add_argument('-txt2')
  parser.add_argument('-labels', default=None)

  parser.add_argument('-out_txt1')
  parser.add_argument('-out_txt2')
  parser.add_argument('-out_labels')
  parser.add_argument('-num_neg', default=1, type=int, help='# of negative examples to generate')

  args = parser.parse_args()
  return args


# Bin -> Representative
def build_binr_map(binr_file, bin_file):
  binr_map = {}
  bin_index = {}
  index_bin = []
  index = 0
  for bin, text in zip(open(bin_file), open(binr_file)):
    bin = bin.strip()
    text = text.strip()
    binr_map[bin] = text
    bin_index[bin] = index
    index_bin.append(bin)
    index += 1
  return binr_map, bin_index, index_bin


# Candidate -> Bin
def build_candidate_bin_map(candidates_file, candidates_bin_file):
  candidate_bin_map = {}

  bins = set()
  for candidate, bin in zip(open(candidates_file), open(candidates_bin_file)):
    candidate = candidate.strip()
    bin = bin.strip()
    candidate_bin_map[candidate] = bin
    bins.add(bin)
  logging.info('#Bins: %d'%len(bins))
  del bins
  return candidate_bin_map


def sample_bin(num_bins, pos_bin):
  while True:
    sampled_bin = np.random.randint(low=0, high=num_bins)
    if sampled_bin != pos_bin:
      return sampled_bin


def process_example(txt1, txt2, num_neg, candidate_bin_map, binr_map, bin_index, index_bin, fw_txt1, fw_txt2, fw_labels):
  txt2 = txt2.strip()
  #Write out with bin representative
  fw_txt1.write(txt1)
  bin_txt2 = candidate_bin_map[txt2]
  pos_bin_index = bin_index[bin_txt2]
  fw_txt2.write(binr_map[bin_txt2] + '\n')
  fw_labels.write('1\n')

  for _ in range(num_neg):
    fw_txt1.write(txt1)
    neg_bin_index = sample_bin(len(bin_index), pos_bin_index)
    neg_bin = index_bin[neg_bin_index]
    fw_txt2.write(binr_map[neg_bin] + '\n')
    fw_labels.write('0\n')



def main():
  args = setup_args()
  logging.info(args)

  np.random.seed(args.seed)
  binr_map, bin_index, index_bin = build_binr_map(args.binr, args.bin)
  logging.info('#Bins: %d'%len(binr_map))

  candidate_bin_map = build_candidate_bin_map(args.candidates, args.candidates_bin)
  logging.info('#Candidates: %d'%len(candidate_bin_map))

  fw_txt1 = open(args.out_txt1, 'w')
  fw_txt2 = open(args.out_txt2, 'w')
  fw_labels = open(args.out_labels, 'w')

  if args.labels is None:
    for txt1, txt2 in zip(open(args.txt1), open(args.txt2)):
      process_example(txt1, txt2, args.num_neg, candidate_bin_map, binr_map, bin_index, index_bin, fw_txt1, fw_txt2, fw_labels)
  else:
    for txt1, txt2, label in zip(open(args.txt1), open(args.txt2), open(args.labels)):
      label = int(label)
      if label == 0:
        continue
      process_example(txt1, txt2, args.num_neg, candidate_bin_map, binr_map, bin_index, index_bin, fw_txt1, fw_txt2, fw_labels)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()