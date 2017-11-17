import logging, argparse

R1 = 0
R2 = 1
R5 = 4

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
    if candidate[1] == index:
      return rank


def process_candidates(candidates, max_rank, datum_num):
  #Sorry, no candidates!
  if len(candidates) == 0:
    return

  #GT is not present in candidates, set rank_gt=k
  if candidates[0][1] != 0:
    return

  candidates = sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
  rank_0 = find_rank_index(0, candidates)
  max_rank[datum_num] = rank_0


def main():
  args = setup_args()
  logging.info(args)

  #Set top-rank to k for each datum
  rank_gt = [args.k for datum in range(args.num_datum)]

  last_datum_num = -1
  candidates = []
  for scores_line, map_line in zip(open(args.scores_file), open(args.map_file)):
    datum_num, index = map_line.split(',')
    datum_num = int(datum_num)

    if datum_num != last_datum_num:
      if last_datum_num >= 0:
        process_candidates(candidates, rank_gt, last_datum_num)
        if rank_gt[last_datum_num] == args.k:
          logging.info('Datum: %d GT absent C: %d'%(datum_num, len(candidates)))
        else:
          logging.info('Datum: %d Rank_0: %d/%d'%(last_datum_num, rank_gt[last_datum_num], len(candidates)-1))
      last_datum_num = datum_num
      candidates = []

    index = int(index)
    score = float(scores_line.strip())
    candidates.append((score, index))

  process_candidates(candidates, rank_gt, last_datum_num)
  logging.info('Datum: %d Rank_0: %d/%d' % (last_datum_num, rank_gt[last_datum_num], len(candidates)))

  num_r1 = 0.0
  num_r2 = 0.0
  num_r5 = 0.0

  gt_present = 0
  gt_absent = 0
  total = 0.0 + len(rank_gt)
  #Given Rank of GT for each datum, it is trivial to compute R@k numbers
  for datum in rank_gt:
    if datum == args.k:
      gt_absent += 1
    else:
      gt_present += 1

    if datum == R1:
      num_r1 += 1
      num_r2 += 1
      num_r5 += 1
    elif datum <= R2:
      num_r2 += 1
      num_r5 += 1
    elif datum <= R5:
      num_r5 += 1


  logging.info('R1:%d R2:%d R@5:%d GT(%d/%d)' % (int(num_r1), int(num_r2), int(num_r5), gt_present, gt_absent))
  logging.info('R@1:%.3f R@2:%.3f R@5:%.3f'%((num_r1/total), (num_r2/total), (num_r5/total)))

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()