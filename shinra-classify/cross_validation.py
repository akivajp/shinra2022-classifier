import json
import random
import argparse
from pathlib import Path

import numpy as np
import chainer
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.datasets import TextDataset, TransformDataset, get_cross_validation_datasets
from chainer.backends import cuda
from chainer.iterators import SerialIterator
from chainer.training import StandardUpdater, Trainer, extensions
from logzero import logger
from tqdm import tqdm

from utils import load_embedding, batch_converter, DatasetReader
from modeling import ENEClassifier


# Unified Memoryを使う
#import cupy as cp
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)


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

    logger.info('loading entity vectors')
    entity_embed = load_embedding(
        args.entity_embed_file, entity2id, args.entity_embed_size, skip_header=True)

    logger.info('loading dataset')
    dataset_reader = DatasetReader(entity2id, feature2id, label2id)
    # dataset = TransformDataset(TextDataset(args.dataset_file), dataset_reader)
    #dataset = list(map(dataset_reader, TextDataset(args.dataset_file)))
    dataset = list(
        tqdm(
            #map(dataset_reader, TextDataset(args.dataset_file))
            map(dataset_reader, open(args.dataset_file)),
        )
    )
    logger.info(f'dataset size: {len(dataset)}')

    cv_datasets = get_cross_validation_datasets(dataset, args.cv_fold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = Path(args.output_dir) / 'prediction.json'

    with open(prediction_file, 'w') as fo:
        for (cv_fold, cv_dataset) in enumerate(cv_datasets, start=1):
            ### release memory
            #import gc
            #model = None
            #optimizer = None
            #updater = None
            #trainer = None
            #batch = None
            #gc.collect()
            ##import cupy as cp
            ##pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
            ##res = pool.free_all_blocks()
            #res = chainer.cuda.memory_pool.free_all_blocks()
            #logger.debug('released memory: %s', res)

            process_fold(
                cv_fold=cv_fold,
                cv_dataset=cv_dataset,
                feature2id=feature2id,
                entity2id=entity2id,
                label2id=label2id,
                id2label=id2label,
                entity_embed=entity_embed,
                fo=fo,
                #train_iterator=train_iterator,
                #eval_iterator=eval_iterator,
            )

            #logger.info(f'*** Cross validation fold {cv_fold} ***')
            #(train_dataset, eval_dataset) = cv_dataset
            #train_iterator = SerialIterator(train_dataset, args.batch_size)
            #eval_iterator = SerialIterator(eval_dataset, args.eval_batch_size,
            #                               repeat=False)

            #model = ENEClassifier(
            #    feature_vocab_size=len(feature2id),
            #    feature_embed_size=len(feature2id),
            #    entity_vocab_size=len(entity2id),
            #    entity_embed_size=args.entity_embed_size,
            #    hidden_size=args.hidden_size,
            #    out_size=len(label2id),
            #    dropout=args.dropout,
            #    feature_initial_embed=np.eye(len(feature2id)),
            #    entity_initial_embed=entity_embed)
            #model.to_device(args.device)

            #optimizer = optimizers.Adam()
            #optimizer.setup(model)

            #model.embed_feature.disable_update()
            #model.embed_entity.disable_update()

            #updater = StandardUpdater(train_iterator, optimizer,
            #                        converter=batch_converter, device=args.device)

            #trainer = Trainer(updater, (args.epoch, 'epoch'), out=args.output_dir)
            #trainer.extend(extensions.LogReport())
            #trainer.extend(extensions.ProgressBar(update_interval=10))
            #trainer.extend(extensions.Evaluator(eval_iterator, model,
            #    converter=batch_converter, device=args.device, eval_func=model.predict))
            #trainer.extend(extensions.PrintReport(
            #    ['epoch', 'main/loss', 'validation/main/loss',
            #    'validation/main/precision', 'validation/main/recall',
            #    'validation/main/f1_score']))

            #trainer.run()

            ## labeling
            #eval_iterator.reset()
            #for batch in tqdm(eval_iterator):
            #    (feature_ids, entity_id, _, infos) = \
            #        batch_converter(batch, args.device, with_info=True)
            #    (preds, probs) = model.predict(feature_ids, entity_id)

            #    for (pred_v, prob_v, info) in zip(preds, probs, infos):
            #        pred_label_ids = np.nonzero(pred_v)[0].tolist()
            #        pred_label_names = []
            #        for label_id in pred_label_ids:
            #            label_prob = prob_v.tolist()[label_id]
            #            label_name = id2label[label_id]
            #            pred_label_names.append({'prob': label_prob, 'ENE': label_name})

            #        ene_annotation = info.get('ene_annotation', dict())
            #        ene_annotation['prediction'] = pred_label_names
            #        out_item = {
            #            'pageid': int(info['pageid']),
            #            'title': info['title'],
            #            'ENEs': ene_annotation,
            #            'gold_ENEs': info['ene_labels']
            #        }
            #        print(json.dumps(out_item, ensure_ascii=False), file=fo)

def process_fold(**kwargs):
    cv_fold = kwargs.get('cv_fold')
    cv_dataset = kwargs.get('cv_dataset')
    feature2id = kwargs['feature2id']
    entity2id = kwargs['entity2id']
    label2id = kwargs['label2id']
    entity_embed = kwargs['entity_embed']
    id2label = kwargs['id2label']
    fo = kwargs['fo']
    #train_iterator = kwargs['train_iterator']
    #eval_iterator = kwargs['eval_iterator']

    logger.info(f'*** Cross validation fold {cv_fold} ***')
    (train_dataset, eval_dataset) = cv_dataset
    train_iterator = SerialIterator(train_dataset, args.batch_size)
    eval_iterator = SerialIterator(eval_dataset, args.eval_batch_size,
                                   repeat=False)

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
    trainer.extend(extensions.Evaluator(eval_iterator, model,
        converter=batch_converter, device=args.device, eval_func=model.predict))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
        'validation/main/precision', 'validation/main/recall',
        'validation/main/f1_score']))

    trainer.run()

    # labeling
    eval_iterator.reset()
    for batch in tqdm(eval_iterator):
        (feature_ids, entity_id, _, infos) = \
            batch_converter(batch, args.device, with_info=True)
        (preds, probs) = model.predict(feature_ids, entity_id)

        for (pred_v, prob_v, info) in zip(preds, probs, infos):
            pred_label_ids = np.nonzero(pred_v)[0].tolist()
            pred_label_names = []
            for label_id in pred_label_ids:
                label_prob = prob_v.tolist()[label_id]
                label_name = id2label[label_id]
                pred_label_names.append({'prob': label_prob, 'ENE': label_name})

            ene_annotation = info.get('ene_annotation', dict())
            ene_annotation['prediction'] = pred_label_names
            out_item = {
                'pageid': int(info['pageid']),
                'title': info['title'],
                'ENEs': ene_annotation,
                'gold_ENEs': info['ene_labels']
            }
            print(json.dumps(out_item, ensure_ascii=False), file=fo)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, required=True,
        help='Dataset file (.json)')
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
    parser.add_argument('--cv_fold', type=int, default=5,
        help='Number of folds for cross validation [5]')
    parser.add_argument('--batch_size', type=int, default=10,
        help='Mini-batch size for training [10]')
    parser.add_argument('--eval_batch_size', type=int, default=1,
        help='Mini-batch size for evaluation [1]')
    parser.add_argument('--epoch', type=int, default=5,
        help='Number of training epochs [5]')
    parser.add_argument('--device', type=str, default='@numpy',
        help='Device specifier (such as "@numpy", "@cupy:0", etc.) [@numpy]')
    args = parser.parse_args()
    main(args)
