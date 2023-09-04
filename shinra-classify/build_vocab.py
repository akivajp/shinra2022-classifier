import json
import argparse
from collections import Counter

from logzero import logger


def main(args):
    entity2id = dict()
    feature_counter = Counter()
    label_counter = Counter()

    logger.info('loading dataset file to count features and labels')
    n_processed = 0
    with open(args.dataset_file) as f:
        for line in f:
            item = json.loads(line)
            assert item['title'] not in entity2id
            entity2id[item['title']] = len(entity2id)
            feature_counter.update(item['features'])
            label_counter.update(item.get('ene_labels', []))

            n_processed += 1
            if n_processed % 10000 == 0:
                logger.info(f'processed: {n_processed}')

    if n_processed % 10000 != 0:
        logger.info(f'processed: {n_processed}')

    logger.info('building vocabulary')
    feature2id = {feature: i for i, (feature, _) in
                  enumerate(feature_counter.most_common(args.feature_vocab_size))}
    label2id = {label: i for i, (label, _) in enumerate(label_counter.most_common())}
    vocab = {'entity2id': entity2id, 'feature2id': feature2id, 'label2id': label2id}
    logger.info(f'feature vocabulary size: {len(feature2id)}')
    logger.info(f'label vocabulary size: {len(label2id)}')

    logger.info('saving vocabulary')
    json.dump(vocab, open(args.output_file, 'w'), ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True,
        help='Dataset file (.json)')
    parser.add_argument('--output_file', type=str, required=True,
        help='Output vocabulary file (.json)')
    parser.add_argument('--feature_vocab_size', type=int,
        help='limit maximum size of feature vocabulary')
    args = parser.parse_args()
    main(args)
