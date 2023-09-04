import json
import gzip
import argparse
from collections import defaultdict, Counter, namedtuple

import MeCab
from logzero import logger


Morph = namedtuple('Morph', ('surface', 'pos1', 'pos2'))
Article = namedtuple('Article',
                     ('title', 'text', 'headings', 'categories', 'upper_categories'))


class MeCabTextAnalyzer(object):
    def __init__(self, dic_path=None):
        if dic_path is not None:
            self.mecab = MeCab.Tagger(f'-d {dic_path}')
        else:
            self.mecab = MeCab.Tagger()

    def morph_tokenize(self, text):
        for line in self.mecab.parse(text).split('\n')[:-2]:
            surface = line.split('\t')[0]
            pos_info = line.split('\t')[1].rstrip().split(',')
            yield Morph(surface=surface, pos1=pos_info[0], pos2=pos_info[1])

    def extract_ngrams(self, text, n=2, unit='word'):
        unigrams = []
        if unit == 'word':
            unigrams = [morph.surface for morph in self.morph_tokenize(text)]
        elif unit == 'pos':
            unigrams = [morph.pos1 for morph in self.morph_tokenize(text)]
        elif unit == 'char':
            unigrams = list(text)
        else:
            raise RuntimeError(f'Invalid unit: {unit}')

        ngrams = []
        for i in range(0, len(unigrams) - n + 1):
            ngrams.append('_'.join(unigrams[i:i + n]))

        return ngrams

    def extract_nouns(self, text):
        nouns = [morph.surface for morph in self.morph_tokenize(text)
                 if morph.pos1 == '名詞']
        return nouns

    def extract_first_sentence(self, text):
        first_sentence = ''
        for morph in self.morph_tokenize(text):
            first_sentence = first_sentence + morph.surface
            if morph.pos1 == '記号' and morph.pos2 == '句点':
                break

        return first_sentence

    def extract_char_types(self, text):
        char_types = []
        for char in text:
            if '\u3040' <= char <= '\u309F':
                char_types.append('ひらがな')
            elif '\u30A0' <= char <= '\u30FF':
                char_types.append('カタカナ')
            elif char.isalpha():
                char_types.append('英字')
            else:
                char_types.append('その他')

        return char_types


class MeCabFeatureExtractor(object):
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def __call__(self, article):
        features = []

        # word [1,2]-grams in the title
        features += [f'TW1_{unigram}' for unigram
                     in self.analyzer.extract_ngrams(article.title, n=1, unit='word')]
        features += [f'TW2_{bigram}' for bigram
                     in self.analyzer.extract_ngrams(article.title, n=2, unit='word')]

        # POS bigrams in the title
        features += [f'TP2_{pos_bigram}' for pos_bigram
                     in self.analyzer.extract_ngrams(article.title, n=2, unit='pos')]

        # last noun in the title
        nouns = self.analyzer.extract_nouns(article.title)
        if nouns:
            features += [f'TLN_{nouns[-1]}']

        # last [1,2,3]-character(s) of the title
        features += [f'TCL1_{article.title[-1]}']
        if len(article.title) >= 2:
            features += [f'TCL2_{article.title[-2:]}']
        if len(article.title) >= 3:
            features += [f'TCL3_{article.title[-3:]}']

        # last character type of the title
        char_types = self.analyzer.extract_char_types(article.title)
        features += [f'TTL1_{char_types[-1]}']

        # last noun in the first sentence
        first_sentence = self.analyzer.extract_first_sentence(article.text)
        nouns_in_first_sentence = self.analyzer.extract_nouns(first_sentence)
        if nouns_in_first_sentence:
            features += [f'SLN_{nouns_in_first_sentence[-1]}']
        else:
            features += ['SLN_']

        # headings
        if article.headings:
            features += [f'H_{heading}' for heading in article.headings]
        else:
            features += ['H_']

        # last nouns of category names
        category_last_nouns = []
        for category in article.categories:
            nouns_in_category = self.analyzer.extract_nouns(category)
            category_last_nouns.extend(nouns_in_category[-1:])

        features += [f'DCLN_{noun}' for noun in category_last_nouns]

        # last nouns of upper category names
        upper_category_last_nouns = []
        for upper_category in article.upper_categories:
            nouns_in_upper_category = self.analyzer.extract_nouns(upper_category)
            upper_category_last_nouns.extend(nouns_in_upper_category[-1:])

        features += [f'UCLN_{noun}' for noun in upper_category_last_nouns]

        return features


def main(args):
    logger.info('initializing a text analyzer')
    analyzer = MeCabTextAnalyzer(args.mecab_dic)

    logger.info('initializing a feature extractor')
    feature_extractor = MeCabFeatureExtractor(analyzer)

    title2annotation = defaultdict(dict)
    if args.annotation_files:
        logger.info('loading Extended Named Entity annotation files')
        for annotation_file in args.annotation_files:
            with open(annotation_file) as f:
                for line in f:
                    item = json.loads(line)
                    title = item['title']  # whitespaces are preserved
                    annotation = item.get('ENEs', dict())
                    title2annotation[title].update(annotation)

    logger.info(f'loaded {len(title2annotation)} articles with ENE annotation')

    logger.info('generating gold ENE labels')
    title2labels = dict()
    annotation_key_counts = Counter()
    for title, annotation in title2annotation.items():
        for anno_key in args.annotation_keys:
            if anno_key in annotation:
                labels = [anno_item['ENE'] for anno_item in annotation[anno_key]]
                annotation_key_counts[anno_key] += 1
                break
        else:
            labels = []

        if labels:
            title2labels[title] = labels

    logger.info('annotation keys used for the gold ENEs:')
    for anno_key in args.annotation_keys:
        logger.info(f'  {anno_key}: {annotation_key_counts[anno_key]:>7d} articles')

    logger.info('loading Wikipedia Cirrussearch general dump file')
    upper_category_mapping = dict()
    n_processed = 0
    with gzip.open(args.cirrus_general_file, 'rt') as f:
        for line in f:
            item = json.loads(line)
            if item.get('namespace') != 14:
                continue

            if item.get('category'):
                upper_category_mapping[item['title']] = item['category']

            n_processed += 1
            if n_processed % 10000 == 0:
                logger.info(f'processed: {n_processed}')

    if n_processed % 10000 != 0:
        logger.info(f'processed: {n_processed}')

    logger.info('loading Wikipedia Cirrussearch content dump file to make a dataset')
    n_processed = 0
    with gzip.open(args.cirrus_content_file, 'rt') as f, \
         open(args.output_file, 'w') as fo:
        pageid = None
        for line in f:
            item = json.loads(line)
            if 'index' in item:
                pageid = item['index']['_id']
                continue

            if item.get('namespace') != 0:
                continue

            title = item['title']  # whitespaces are preserved
            text = item.get('text', '')
            first_sentence = analyzer.extract_first_sentence(text)
            headings = item.get('heading', [])
            categories = item.get('category', [])
            upper_categories = []
            for category in categories:
                upper_categories.extend(upper_category_mapping.get(category, []))

            ene_labels = title2labels.get(title)
            ene_annotation = title2annotation.get(title)

            if args.labeled_only and not ene_labels:
                continue
            if args.unlabeled_only and ene_labels:
                continue

            article = Article(title, text, headings, categories, upper_categories)
            features = feature_extractor(article)

            if args.inject_datestamp:
                title = f'{title}##{args.inject_datestamp}'
            out_item = {
                'pageid': pageid,
                'title': title,
                'first_sentence': first_sentence,
                'features': features
            }
            if ene_labels:
                out_item['ene_labels'] = ene_labels
            if ene_annotation:
                out_item['ene_annotation'] = ene_annotation

            print(json.dumps(out_item, ensure_ascii=False), file=fo)
            n_processed += 1

            if n_processed <= 10:
                logger.info('*** Example ***')
                logger.info(f'  pageid: {pageid}')
                logger.info(f'  title: {title}')
                logger.info(f'  first_sentence: {first_sentence}')
                logger.info(f'  features: {features}')
                if ene_labels:
                    logger.info(f'  ene_labels: {ene_labels}')
                if ene_annotation:
                    logger.info(f'  ene_annotation: {ene_annotation}')

            if n_processed % 10000 == 0:
                logger.info(f'processed: {n_processed}')

    if n_processed % 10000 != 0:
        logger.info(f'processed: {n_processed}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cirrus_content_file', type=str, required=True,
        help='Wikipedia Cirrussearch content dump file (.json.gz)')
    parser.add_argument('--cirrus_general_file', type=str, required=True,
        help='Wikipedia Cirrussearch general dump file (.json.gz)')
    parser.add_argument('--output_file', type=str, required=True,
        help='Output file path (.json)')
    parser.add_argument('--annotation_files', type=str, nargs='*',
        help='Extended Named Entity annotation file (.json)')
    parser.add_argument('--annotation_keys', type=str, nargs='*',
        help='annotation keys used in annotation_file (whitespace-separated list)')
    parser.add_argument('--mecab_dic', type=str,
        help='Path to the directory of Mecab dictionary')
    parser.add_argument('--labeled_only', action='store_true',
        help='Output only labeled titles')
    parser.add_argument('--unlabeled_only', action='store_true',
        help='Output only not labeled titles')
    parser.add_argument('--inject-datestamp', type=str,
        help='Append date stamp and express entity name as "ENTITY##DATESTAMP')
    args = parser.parse_args()
    main(args)
