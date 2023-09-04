import json
import random
import argparse

import numpy as np
import chainer
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.datasets import TextDataset, TransformDataset, get_cross_validation_datasets
from chainer.backends import cuda
from chainer.iterators import SerialIterator
from chainer.training import StandardUpdater, Trainer, extensions
from logzero import logger

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
    logger.info(f'feature vocabulary size: {len(feature2id)}')
    logger.info(f'label vocabulary size: {len(label2id)}')

    logger.info('loading entity vectors')
    entity_embed = load_embedding(
        args.entity_embed_file, entity2id, args.entity_embed_size, skip_header=True)

    logger.info('loading dataset')
    dataset_reader = DatasetReader(entity2id, feature2id, label2id)
    train_dataset = TransformDataset(TextDataset(args.train_file), dataset_reader)
    train_iterator = SerialIterator(train_dataset, args.batch_size)
    if args.eval_file:
        eval_dataset = list(map(dataset_reader, TextDataset(args.eval_file)))
        eval_iterator = SerialIterator(eval_dataset, 1, repeat=False)
    else:
        eval_iterator = None

    model = ENEClassifier(
        feature_vocab_size=len(feature2id),
        feature_embed_size=len(feature2id),
        entity_vocab_size=len(entity2id),
        entity_embed_size=args.entity_embed_size,
        hidden_size=args.hidden_size,
        out_size=len(label2id),
        dropout=args.dropout,
        feature_initial_embed=np.eye(len(feature2id)),
        entity_initial_embed=entity_embed)
    model.to_device(args.device)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    model.embed_feature.disable_update()
    model.embed_entity.disable_update()

    updater = StandardUpdater(train_iterator, optimizer,
                                converter=batch_converter, device=args.device)

    trainer = Trainer(updater, (args.epoch, 'epoch'), out=args.output_dir)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.snapshot_object(model,
        filename='epoch-{.updater.epoch:03d}.model'), trigger=(5, 'epoch'))
    if eval_iterator is not None:
        trainer.extend(extensions.Evaluator(eval_iterator, model,
            converter=batch_converter, device=args.device, eval_func=model.predict))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'validation/main/precision', 'validation/main/recall',
             'validation/main/f1_score']))
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))

    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True,
        help='Dataset file for training (.json)')
    parser.add_argument('--eval_file', type=str,
        help='Dataset file for evaluation (.json)')
    parser.add_argument('--vocab_file', type=str, required=True,
        help='Vocabulary file (.json)')
    parser.add_argument('--entity_embed_file', type=str, required=True,
        help='Entity embedding file (.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Output directory')
    parser.add_argument('--entity_embed_size', type=int, required=True,
        help='Entity embedding size (number of dimentions)')
    parser.add_argument('--hidden_size', type=int, default=200,
        help='Hidden vector size [200]')
    parser.add_argument('--dropout', type=float, default=0.0,
        help='Hidden vector size [0.0]')
    parser.add_argument('--batch_size', type=int, default=10,
        help='Mini-batch size [10]')
    parser.add_argument('--epoch', type=int, default=5,
        help='Number of training epochs [5]')
    parser.add_argument('--device', type=str, default='@numpy',
        help='Device specifier (such as "@numpy", "@cupy:0", etc.) [@numpy]')
    args = parser.parse_args()
    main(args)
