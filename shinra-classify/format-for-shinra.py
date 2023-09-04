#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys

from collections import OrderedDict

from tqdm import tqdm
from logzero import logger

def main(args):
    id2records = dict()
    max_id = 0
    definition = dict()
    if args.definition:
        with open(args.definition) as f:
            for line in tqdm(f, desc='loading definition'):
                rec = json.loads(line)
                #id = rec['id']
                id = rec['ENE_id']
                definition[id] = rec
    with open(args.input_file) as f:
        #oProgress = tqdm(desc = 'output')
        for i, line in enumerate(tqdm(f, desc = 'input')):
            #if i > 10:
            #    break
            rec = json.loads(line)
            #logger.debug('rec: %s', rec)
            title = rec['title']
            pageid = int(rec['pageid'])
            fields = title.split('##')
            if len(fields) != 2:
                logger.error('line number: %s', i+1)
                logger.error('line: %s', line)
                logger.error('invalid tagged title: %s', title)
                return False
            main_title, tag = fields
            if tag == args.tag:
                rec['title'] = main_title
                #print(json.dumps(rec))
                #oProgress.update(1)
                max_id = max(max_id, pageid)
                id2records[pageid] = rec
    oProgress = tqdm(desc = 'output')
    for i in tqdm(range(max_id+1), desc='id loop'):
        #logger.debug('i: %s', i)
        #logger.debug('i in id2records: %s', i in id2records)
        if i in id2records:
            rec = id2records[i]
            for key, probs in rec['ENEs'].items():
                for prob in probs:
                    ene = prob['ENE']
                    if ene in definition:
                        defRec = definition[ene]
                        prob['ENE_en'] = defRec['name']['en']
                        prob['ENE_ja'] = defRec['name']['ja']
            #print(json.dumps(rec, ensure_ascii=False))
            outRec = OrderedDict()
            outRec['page_id'] = rec['pageid']
            outRec['title'] = rec['title']
            outRec['ENEs'] = rec['ENEs']
            print(json.dumps(outRec, ensure_ascii=False))
            oProgress.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-t', '--tag', type=str, required=True)
    parser.add_argument('-d', '--definition', type=str)
    args = parser.parse_args()
    main(args)
