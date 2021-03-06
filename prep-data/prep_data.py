'''
Separate data into three parallel files: txt1, txt2 and labels
Apllied to train, valid, and test
'''
import logging, argparse, csv
import numpy as np
POS_LABEL = 1
NEG_LABEL = 0

NUM_COLS_TRAIN = 3
NUM_COLS_TEST = 11

#Train data has three fields Context,Utterance,Label
#Valid and Test data has Context,Ground Truth Utterance,Distractor0,..., DistractorN-1
def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-train', default=False, action='store_true')
  parser.add_argument('-siamese_valid', default=False, action='store_true')
  parser.add_argument('-only_positive', default=False, action='store_true')
  parser.add_argument('-preserve_case', default=False, action='store_true')
  parser.add_argument('-input_csv', default=None)
  parser.add_argument('-out_txt1', default=None)
  parser.add_argument('-out_txt2', default=None)
  parser.add_argument('-out_labels', default=None)
  parser.add_argument('-seed', default=1543, type=int)
  args = parser.parse_args()
  return args


def process_training_data(input_csv, out_txt1, out_txt2, out_labels):
  fw_txt1 = open(out_txt1, 'w')
  fw_txt2 = open(out_txt2, 'w')
  fw_labels = open(out_labels, 'w')

  num_rows_written = 0
  with open(input_csv) as fr:
    reader = csv.reader(fr)
    #Skip header
    next(reader, None)

    for row in reader:
      assert len(row) == NUM_COLS_TRAIN
      fw_txt1.write('%s\n'%row[0].lower().strip())
      fw_txt2.write('%s\n'%row[1].lower().strip())
      fw_labels.write('%s\n'%row[2].lower().strip())
      num_rows_written += 1

      if num_rows_written % 100000 == 0:
        logging.info('Input: %s IP Wrote %d rows'%(input_csv, num_rows_written))

  logging.info('Input: %s Done Wrote %d rows'%(input_csv, num_rows_written))


#Siamese network needs a validation data with equal positive and negative labels
def process_siamese_valid_data(input_csv, out_txt1, out_txt2, out_labels):
  fw_txt1 = open(out_txt1, 'w')
  fw_txt2 = open(out_txt2, 'w')
  fw_labels = open(out_labels, 'w')

  num_rows_written = 0
  with open(input_csv) as fr:
    reader = csv.reader(fr)
    # Skip header
    next(reader, None)

    for row in reader:
      assert len(row) == NUM_COLS_TEST

      fw_txt1.write('%s\n'%row[0].lower().strip())
      fw_txt2.write('%s\n' % row[1].lower().strip())
      fw_labels.write('%d\n'%POS_LABEL)

      neg_index = np.random.randint(2, NUM_COLS_TEST)
      fw_txt1.write('%s\n' % row[0].lower().strip())
      fw_txt2.write('%s\n' % row[neg_index].lower().strip())
      fw_labels.write('%d\n' % NEG_LABEL)

      num_rows_written += 1
      if num_rows_written % 100000 == 0:
        logging.info('Input: %s IP Wrote %d rows'%(input_csv, num_rows_written))
  logging.info('Input: %s Done Wrote %d rows'%(input_csv, num_rows_written))


#Write only GT data
def process_positive_data(input_csv, out_txt1, out_txt2):
  fw_txt1 = open(out_txt1, 'w')
  fw_txt2 = open(out_txt2, 'w')

  with open(input_csv) as fr:
    reader = csv.reader(fr)
    # Skip header
    next(reader, None)

    for row in reader:
      assert len(row) == NUM_COLS_TEST
      fw_txt1.write('%s\n' % row[0].lower().strip())
      fw_txt2.write('%s\n' % row[1].lower().strip())


def main():
  args = setup_args()
  logging.info(args)

  np.random.seed(args.seed)
  if args.train:
    process_training_data(args.input_csv, args.out_txt1, args.out_txt2, args.out_labels)
  elif args.siamese_valid:
    process_siamese_valid_data(args.input_csv, args.out_txt1, args.out_txt2, args.out_labels)
  elif args.only_positive:
    process_positive_data(args.input_csv, args.out_txt1, args.out_txt2)

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()