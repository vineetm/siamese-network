import tensorflow as tf
import argparse, os, csv

logging = tf.logging
logging.set_verbosity(logging.INFO)

CSV = 'csv'
GT = 'gt'
DISTRACTOR = 'd'
NUM_DISTRCACTORS = 9

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('raw_data_dir', help='Raw Ubuntu data directory')
  parser.add_argument('out_dir', help='Data directory')

  parser.add_argument('-train', default='train')
  parser.add_argument('-valid', default='valid')
  parser.add_argument('-test',  default='test')

  #Used for training
  parser.add_argument('-txt1', default='txt1')
  parser.add_argument('-txt2', default='txt2')
  parser.add_argument('-labels', default='labels')

  args = parser.parse_args()
  return args


def process_train_data(data_dir, out_dir, train_suffix, text1, text2, labels):
  train_src_file = os.path.join(data_dir, '%s.%s'%(train_suffix, CSV))

  logging.info('Reading train file: %s'%train_src_file)
  fw_txt1 = open(os.path.join(out_dir, '%s.%s' % (train_suffix, text1)), 'w')
  fw_txt2 = open(os.path.join(out_dir, '%s.%s' % (train_suffix, text2)), 'w')
  fw_labels = open(os.path.join(out_dir, '%s.%s' % (train_suffix, labels)), 'w')

  with open(train_src_file) as fr:
  #Get CSV Reader, and skip header
    reader = csv.reader(fr)
    next(reader)

    for row in reader:
      assert len(row) == 3
      text1, text2, label = row

      #Text1 is a combination of previous utterances
      fw_txt1.write('%s\n'%(' '.join(text1.split(','))))

      fw_txt2.write('%s\n'%text2)
      fw_labels.write('%s\n'%label)

  fw_txt1.close()
  fw_txt2.close()
  fw_labels.close()


def process_valid_data(data_dir, out_dir, valid_suffix, text1, text2):
  valid_src_file = os.path.join(data_dir, '%s.%s' % (valid_suffix, CSV))
  logging.info('Reading valid file: %s'%valid_src_file)

  fw_txt1    = open(os.path.join(out_dir, '%s.%s' % (valid_suffix, text1)), 'w')
  fw_txt2_gt = open(os.path.join(out_dir, '%s.%s.%s'%(valid_suffix, text2, GT)), 'w')

  fw_txt2_d = []
  for di in range(NUM_DISTRCACTORS):
    fw = open(os.path.join(out_dir, '%s.%s.%s%d'%(valid_suffix, text2, DISTRACTOR, di)), 'w')
    fw_txt2_d.append(fw)

  distractor_start_index = 2
  with open(valid_src_file) as fr:
    reader = csv.reader(fr)
    next(reader)

    for row in reader:
      assert len(row) == 11

      fw_txt1.write('%s\n' % (' '.join(row[0].split(','))))
      fw_txt2_gt.write('%s\n'%row[1])

      for di, distractor in enumerate(row[distractor_start_index:]):
        fw_txt2_d[di].write('%s\n'%distractor)


def main():
  args = setup_args()
  logging.info(args)

  process_train_data(args.raw_data_dir, args.out_dir, args.train,
                     args.txt1, args.txt2, args.labels)

  process_valid_data(args.raw_data_dir, args.out_dir, args.valid,
                     args.txt1, args.txt2)


if __name__ == '__main__':
  main()