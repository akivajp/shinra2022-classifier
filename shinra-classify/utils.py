import json
import random

import numpy as np
import chainer
from chainer.backends import cuda
from chainer.dataset import concat_examples
from logzero import logger


from tqdm import tqdm


random.seed(0)
np.random.seed(0)
if cuda.available:
    cuda.cupy.random.seed(0)


def load_embedding(embed_file, x2id, embed_size, skip_header=False):
    embed_matrix = np.zeros((len(x2id), embed_size), dtype='f')
    with open(embed_file) as f:
        if skip_header:
            f.readline()

        #for line in f:
        for line in tqdm(f):
            (x_token, *v) = line.rstrip('\n').split(' ')
            x_token = x_token.replace('_', ' ')
            if x_token in x2id:
                x_id = x2id[x_token]
                embed_matrix[x_id] = v

    return embed_matrix


def batch_converter(batch, device, with_info=False):
    (data_items, info_items) = zip(*batch)
    data_items = concat_examples(data_items, device, padding=-1)
    if with_info:
        return (*data_items, info_items)
    else:
        return (*data_items,)


class DatasetReader(object):
    def __init__(self, entity2id, feature2id, label2id):
        self.entity2id = entity2id
        self.feature2id = feature2id
        self.label2id = label2id

    def __call__(self, line):
        item = json.loads(line)
        #entity_id = self.entity2id[item['title']]
        entity_id = self.entity2id.get(item['title'], -1)
        feature_ids = [self.feature2id[feature] for feature in set(item['features'])
                       if feature in self.feature2id]
        label_ids = [0] * len(self.label2id)
        for label in item.get('ene_labels', []):
            #label_ids[self.label2id[label]] = 1
            if label in self.label2id:
                label_ids[self.label2id[label]] = 1

        data_item = (np.array(feature_ids, dtype='i'),
                     np.array(entity_id, dtype='i'),
                     np.array(label_ids, dtype='i'))
        info_item = item
        return (data_item, info_item)
