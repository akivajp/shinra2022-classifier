import json
import random
import argparse

import numpy as np
import chainer
from chainer import optimizers, serializers
from chainer.dataset import concat_examples
from chainer.datasets import TextDataset, TransformDataset, get_cross_validation_datasets
from chainer.backends import cuda
from chainer.iterators import SerialIterator
from chainer.training import StandardUpdater, Trainer, extensions
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from logzero import logger
from tqdm import tqdm

from utils import load_embedding, batch_converter, DatasetReader
from modeling import ENEClassifier


random.seed(0)
np.random.seed(0)
if cuda.available:
    cuda.cupy.random.seed(0)


def main(args):
    logger.info('loading vocabulary')
    vocab = json.load(open(args.vocab_file))
    entity2id = vocab['entity2id']
    feature2id = vocab['feature2id']
    label2id = vocab['label2id']
    id2label = {i: label for label, i in label2id.items()}
    logger.info(f'feature vocabulary size: {len(feature2id)}')
    logger.info(f'label vocabulary size: {len(label2id)}')

    logger.info('loading dataset')
    dataset_reader = DatasetReader(entity2id, feature2id, label2id)
    dataset = TransformDataset(TextDataset(args.dataset_file), dataset_reader)
    dataset_iterator = SerialIterator(dataset, args.batch_size,
                                      repeat=False, shuffle=False)

    model = ENEClassifier(
        feature_vocab_size=len(feature2id),
        feature_embed_size=len(feature2id),
        entity_vocab_size=len(entity2id),
        entity_embed_size=args.entity_embed_size,
        hidden_size=args.hidden_size,
        out_size=len(label2id),
        dropout=args.dropout,
        #nbest=args.nbest,
    )
    serializers.load_npz(args.model_file, model)
    model.to_device(args.device)

    pred_vectors, ref_vectors = [], []
    with open(args.output_file, 'w') as fo:
        for batch in tqdm(dataset_iterator):
            (feature_ids, entity_id, label_ids, infos) = \
                batch_converter(batch, args.device, with_info=True)
            #(preds, probs) = model.predict(feature_ids, entity_id)
            (preds, probs) = model.predict(feature_ids, entity_id, nbest=args.nbest)

            for (pred_v, prob_v, label_v, info) in zip(preds, probs, label_ids, infos):
                pred_label_ids = np.nonzero(pred_v)[0].tolist()
                pred_label_names = []
                for label_id in pred_label_ids:
                    label_prob = prob_v.tolist()[label_id]
                    label_name = id2label[label_id]
                    pred_label_names.append({'prob': label_prob, 'ENE': label_name})

                pred_label_names = sorted(pred_label_names, key=lambda e: e['prob'], reverse=True)
                ene_annotation = info.get('ene_annotation', dict())
                ene_annotation[args.output_key] = pred_label_names
                out_item = {
                    'pageid': int(info['pageid']),
                    'title': info['title'],
                    'ENEs': ene_annotation
                }
                if args.output_gold_labels and 'ene_labels' in info:
                    out_item['gold_ENEs'] = info['ene_labels']

                print(json.dumps(out_item, ensure_ascii=False), file=fo)

                if args.calc_scores:
                    if np.any(label_v):
                        pred_vectors.append(cuda.to_cpu(pred_v))
                        ref_vectors.append(cuda.to_cpu(label_v))

    #logger.debug(pred_vectors)
    if args.calc_scores:
        if pred_vectors:
            pred_matrix = np.vstack(pred_vectors)
            ref_matrix = np.vstack(ref_vectors)

            print('Instance-wise averaged metrics:')
            (precision, recall, f1_score, _) = \
                precision_recall_fscore_support(ref_matrix, pred_matrix, average='samples')
            print(f'  Precision: {precision:.2%}')
            print(f'  Recall:    {recall:.2%}')
            print(f'  F1 score:  {f1_score:.2%}')
            print('')

            print('Class-wise averaged metrics (macro average):')
            (precision, recall, f1_score, _) = \
                precision_recall_fscore_support(ref_matrix, pred_matrix, average='macro')
            print(f'  Precision: {precision:.2%}')
            print(f'  Recall:    {recall:.2%}')
            print(f'  F1 score:  {f1_score:.2%}')
            print('')

            print('Metric for each class:')
            (ps, rs, fs, ss) = \
                precision_recall_fscore_support(ref_matrix, pred_matrix, average=None)
            print('Precision\tFrequency\tRecall\tF1 Score')
            for i, (p, r, f, s) in enumerate(zip(ps, rs, fs, ss)):
                label_name = id2label[i]
                print(f'{label_name}\t{s}\t{p:.4f}\t{r:.4f}\t{f:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True,
        help='Dataset file to be labeled (.json)')
    parser.add_argument('--vocab_file', type=str, required=True,
        help='Vocabulary file (.json)')
    parser.add_argument('--model_file', type=str, required=True,
        help='Entity embedding file (.txt)')
    parser.add_argument('--output_file', type=str, required=True,
        help='Output file')
    parser.add_argument('--output_key', type=str, default='AUTO.TOHOKU.201906',
        help='Dictionary key for the models\'s prediction (.json)')
    parser.add_argument('--entity_embed_size', type=int, required=True,
        help='Entity embedding size (number of dimentions)')
    parser.add_argument('--hidden_size', type=int, default=200,
        help='Hidden vector size [200]')
    parser.add_argument('--dropout', type=float, default=0.0,
        help='Dropout ratio [0.0]')
    parser.add_argument('--batch_size', type=int, default=10,
        help='Mini-batch size [10]')
    parser.add_argument('--output_gold_labels', action='store_true',
        help='Output annotated gold ENE labels along with predictions')
    parser.add_argument('--device', type=str, default='@numpy',
        help='Device specifier (such as "@numpy", "@cupy:0", etc.) [@numpy]')
    parser.add_argument('--nbest', type=int, default=1,
        help='Minimal number of predicted candidates to write out')
    from distutils.util import strtobool
    parser.add_argument('--calc_scores', type=strtobool, default=True,
        help='Calculate scores')
    args = parser.parse_args()
    logger.debug('args: %s', args)
    main(args)
