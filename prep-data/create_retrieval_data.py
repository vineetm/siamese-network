'''
Create retrieval data for R@k tasks
'''
import argparse, logging
import numpy as np
import pickle as pkl


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-candidates_pkl')
  parser.add_argument('-input_txt1')
  #GT file, outputted as is
  parser.add_argument('-input_txt2')

  parser.add_argument('-output_txt1')
  parser.add_argument('-output_txt2')
  parser.add_argument('-output_map')

  parser.add_argument('-random_seed', default=1543, type=int)
  parser.add_argument('-k', default=1000, type=int)

  args = parser.parse_args()
  return args


def get_k_random_values(k, txt2, candidates):
  txt2 = txt2.strip()
  random_values = []
  while True:
    rv = np.random.randint(0, len(candidates))
    if rv in random_values:
      continue

    if candidates[rv] == txt2:
      continue

    random_values.append(rv)
    if len(random_values) == k:
      return random_values


def main():
  args = setup_args()
  logging.info(args)

  np.random.seed(args.random_seed)
  fw_txt1 = open(args.output_txt1, 'w')
  fw_txt2 = open(args.output_txt2, 'w')
  fw_map = open(args.output_map, 'w')

  with open(args.candidates_pkl, 'rb') as fr:
    candidates = pkl.load(fr)
  logging.info('Num_Candidates %d'%len(candidates))

  datum = 0

  for txt1, txt2 in zip(open(args.input_txt1), open(args.txt2)):
    fw_txt1.write(txt1)
    fw_txt2.write(txt2)
    fw_map.write('%d,%d' % (datum, 0))
    rvs = get_k_random_values(args.k, txt2, candidates)
    for index, cnum in enumerate(rvs):
      fw_txt1.write(txt1)
      fw_txt2.write('%s\n'%candidates[cnum])
      fw_map.write('%d,%d'%(datum, index+1))

    datum += 1
    if datum % 10 == 0:
      logging.info('Processed: %d'%datum)


if __name__ == '__main__':
  main()

