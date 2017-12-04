import logging, argparse
'''
Separate last turn from remaining data
'''
EOT = '__eot__'

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-input')
  parser.add_argument('-context')
  parser.add_argument('-last_turn')
  args = parser.parse_args()
  return args

def get_tokens_with_eot(parts):
  all_tokens = []
  for part in parts:
    all_tokens.extend(part.strip().split())
    all_tokens.append(EOT)
  return all_tokens


def main():
  args = setup_args()
  logging.info(args)

  fw_context = open(args.context, 'w')
  fw_last_turn = open(args.last_turn, 'w')

  for line in open(args.input):
    parts = line.split(EOT)
    #Last part is empty
    parts = parts[:-1]

    fw_last_turn.write('%s\n'%' '.join(get_tokens_with_eot([parts[-1]])))
    fw_context.write('%s\n'%' '.join(get_tokens_with_eot(parts[:-1])))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  main()