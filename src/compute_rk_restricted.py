import logging, argparse

#Candidates to check
RK = [1, 2, 5]

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-scores_file')
  parser.add_argument('-map_file')
  parser.add_argument('-k', type=int, default=10)
  parser.add_argument('-num_datum', default=19560, type=int)
  args = parser.parse_args()
  return args


def find_rank_index(index, candidates):
  for rank, candidate in enumerate(candidates):
    if candidate[1] == 0:
      return rank


def process_max_rank(candidates, max_rank, datum_num):
  if len(candidates) == 0:
    return

  if candidates[0][1] != 0:
    return

  candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
  rank_0 = find_rank_index(0, candidates)
  max_rank[datum_num] = rank_0


def main():
  args = setup_args()
  logging.info(args)

  #Set top-rank to k for each datum
  max_rank = [args.k for datum in range(args.num_datum)]

  last_datum_num = -1
  candidates = []
  for scores_line, map_line in zip(open(args.scores_file), open(args.map_file)):
    datum_num, index = map_line.split(',')
    datum_num = int(datum_num)

    if datum_num != last_datum_num:
      if last_datum_num >= 0:
        process_max_rank(candidates, max_rank, last_datum_num)
        logging.info('Datum: %d Rank_0: %d/%d'%(last_datum_num, max_rank[last_datum_num], len(candidates)))
      last_datum_num = datum_num
      candidates = []

    index = int(index)
    score = float(scores_line.strip())
    candidates.append((score, index))

  process_max_rank(candidates, max_rank, last_datum_num)
  logging.info('Datum: %d Rank_0: %d/%d' % (last_datum_num, max_rank[last_datum_num], len(candidates)))

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()