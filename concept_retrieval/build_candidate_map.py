import logging, argparse
import pickle as pkl

def build_word2cluster_map(word_clusters):
    word2cluster = {}
    clusters = []
    for index, cluster in enumerate(open(word_clusters)):
        members = cluster.strip().split()
        for member in members:
            if member in word2cluster:
                continue
            word2cluster[member] = index
        clusters.append(members)
    return word2cluster, clusters


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-word_clusters')
    parser.add_argument('-candidates')

    parser.add_argument('-cluster_map')
    parser.add_argument('-cluster_count')

    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    logging.info(args)

    word2cluster, clusters = build_word2cluster_map(args.word_clusters)
    logging.info(f'Clusters {len(clusters)}')

    cluster_map = [set() for _ in range(len(clusters))]
    for index, candidate in enumerate(open(args.candidates)):
        cluster_present = [word2cluster[word] for word in candidate.strip().split() if word in word2cluster]

        for cluster in set(cluster_present):
            cluster_map[cluster].add(str(index))

        if index % 10000 == 0:
            logging.info(index)

    fw_cl_map = open(args.cluster_map, 'w')
    fw_cl_count = open(args.cluster_count, 'w')

    for k in range(len(clusters)):
        fw_cl_map.write(f'''{' '.join(cluster_map[k])}\n''')
        fw_cl_count.write(f'{len(cluster_map[k])}\n')

    with open(f'{args.cluster_map}.pkl', 'wb') as fw:
        pkl.dump(cluster_map, fw)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()