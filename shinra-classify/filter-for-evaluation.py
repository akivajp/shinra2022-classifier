#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

from logzero import logger
from tqdm import tqdm

prediction_path = sys.argv[1]
evaluation_path = sys.argv[2]

prediction_by_pageid = {}
with open(prediction_path, 'r') as f:
    for i, line in enumerate(tqdm(f)):
        #if i > 1000:
        #    break
        rec = json.loads(line)
        page_id = str(rec['page_id'])
        prediction_by_pageid[page_id] = rec

with open(evaluation_path, 'r') as f:
    for i, line in enumerate(tqdm(f)):
        #if i > 1000:
        #    break
        rec = json.loads(line)
        page_id = str(rec['page_id'])
        if page_id in prediction_by_pageid:
            prediction_rec = prediction_by_pageid[page_id]
            #enes = prediction_rec['ENEs']
            #for label, predictions in enes.items():
            #    for prediction in predictions:
            #        ene_name_ja = prediction['ENE']
            #        #if ene_name_ja == '人名:キャラクター名':
            #        #    ene_name_ja = 'キャラクター名'
            #        if ene_name_ja.find(':') > 0:
            #            # '人名:キャラクター名' -> 'キャラクター名'
            #            ene_name_ja = ene_name_ja.split(':')[-1]
            #        if ene_name_ja == '資格試験名':
            #            # データの不具合。最新版は修正されているが、再学習が間に合わないためここで書き換え処理
            #            ene_name_ja = '試験名'
            #        prob = prediction['prob']
            #        #logger.debug('ENE name ja: %s', ene_name_ja)
            #        #if ene_name_ja not in definition_by_name_ja:
            #        #    logger.debug('prediction_rec: %s', prediction_rec)
            #        ene_id = definition_by_name_ja[ene_name_ja]['ENE_id']
            #        prediction['ENE'] = ene_id
            rec['ENEs'] = prediction_rec['ENEs']
            print(json.dumps(rec, ensure_ascii=False))
