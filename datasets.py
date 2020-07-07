"""
数据集
"""

import os
import copy
import glob
import codecs
import pickle
import random
from typing import Dict, List, Any, Set
from itertools import chain
from collections import defaultdict

import torch

from nmnlp.common.constant import KEY_TRAIN, KEY_DEV, KEY_TEST, PRETRAIN_POSTFIX
from nmnlp.core.dataset import DataSet
from nmnlp.data.conll import ConlluDataset

from translate import Translate

try:
    import xml.etree.cElementTree as et
except ImportError:
    import xml.etree.ElementTree as et

translate = Translate()


class STREUSLEDataset(DataSet):
    """ super-sense dataset """
    index_fields = ("words", "upostag", "labels")

    def __init__(self,
                 data: List,
                 tokenizer: Any = None,
                 pretrained_fields: Set[str] = ()):
        super().__init__(data, tokenizer, pretrained_fields)

    @classmethod
    def build(cls,
              data_dir: str,
              kind: str = KEY_TRAIN,
              tokenizer: Any = None,
              pretrained_fields: Set[str] = ()):
        path = glob.glob(f"{data_dir}/*/*{kind}.conllulex")[0]
        dataset = cls(list(), tokenizer, pretrained_fields)

        with codecs.open(path, mode='r', encoding='UTF-8') as file:
            sentence = list()
            for line in chain(file, [""]):
                line = line.strip()
                if not line and sentence:
                    if len(sentence[0]) > 10:
                        dataset.text_to_instance(sentence)
                    sentence = list()
                elif line.startswith('#'):
                    continue
                else:
                    line = line.split('\t')
                    try:
                        line[0] = int(line[0])
                        sentence.append(line)
                    except ValueError:
                        continue

        return dataset

    def text_to_instance(self, sentence):
        ins = {'words': list(), 'upostag': list(), 'sent': list(), 'labels': list()}
        pieces, mwe = dict(), dict()
        # print('\n')

        padding_len = len(sentence[0]) - 4
        sentence.insert(0, [0, '[CLS]', '[CLS]', 'X'] + ['_'] * padding_len)
        sentence.append([len(sentence), '[SEP]', '[SEP]', 'X'] + ['_'] * padding_len)

        for i, row in enumerate(sentence):
            # print('\t'.join(row[10:]))
            ins['sent'].append(row[1])
            ins['upostag'].append(row[3])
            if self.tokenizer is not None:
                piece = self.tokenizer.tokenize(row[1])
                if len(piece) > 0:
                    ins['words'].append(piece[0])
                    if len(piece) > 1:
                        pieces[i] = [self.tokenizer.vocab[p] for p in piece]
            else:
                ins['words'].append(row[1].lower())
            if row[13] != '_':
                label = row[13].lower()
                if row[10] == '_':
                    ins['labels'].append(f"O_{label}")
                else:
                    mwe[row[10].split(':')[0]] = label
                    ins['labels'].append(f"B_{label}")
            else:
                ins['labels'].append('_')

        for i, row in enumerate(sentence):
            if row[10] == '_':
                continue
            mwe_id, no = row[10].split(':')
            if no != '1' and mwe_id in mwe:
                ins['labels'][i] = f"I_{mwe[mwe_id]}"

        ins['word_pieces'] = pieces
        if len(self.pretrained_fields) > 0:
            ins["words" + PRETRAIN_POSTFIX] = copy.deepcopy(ins['words'])

        self.data.append(ins)

    def collate_fn(self, batch):
        ids_sorted = sorted(
            range(len(batch)), key=lambda i: len(batch[i]['words']), reverse=True)

        max_len = len(batch[ids_sorted[0]]['words'])
        result = defaultdict(lambda: torch.zeros(len(batch), max_len, dtype=torch.long))
        result['seq_lens'] = list()
        result['sentences'] = list()
        result['word_pieces'] = dict()

        for i, origin in zip(range(len(batch)), ids_sorted):
            seq_len = len(batch[origin]['words'])
            result['seq_lens'].append(seq_len)
            result['sentences'].append(batch[origin]['sent'])
            result['mask'][i, :seq_len] = 1
            for key in ('words', 'upostag', 'labels'):
                result[key][i, :seq_len] = torch.LongTensor(batch[origin][key])
            for key in self.pretrained_fields:
                result[key][i, :seq_len] = torch.LongTensor(batch[origin][key])
            for w, piece in batch[origin]['word_pieces'].items():
                result['word_pieces'][(i, w)] = torch.LongTensor(piece)
            # if (result['words'][i] == 100).long().sum() > seq_len // 3:
            #     print(batch[0]['metadata']['lang'], ':', batch[origin]['form'])

        return result


LABEL = defaultdict(int)


class SRLDataset(DataSet):
    """ SRL dataset """
    index_fields = ("words", "upostag", "labels")

    def __init__(self,
                 data: List,
                 tokenizer: Any = None,
                 pretrained_fields: Set[str] = ()):
        super().__init__(data, tokenizer, pretrained_fields)

    @classmethod
    def build(cls,
              data_dir: str,
              kind: str = KEY_TRAIN,
              lang: str = 'de',
              tokenizer: Any = None,
              pretrained_fields: Set[str] = ()):
        path = glob.glob(f"{data_dir}/*{kind}.conllu")[0]
        dataset = cls(list(), tokenizer, pretrained_fields)

        with codecs.open(path, mode='r', encoding='UTF-8') as file:
            sentence = list()
            for line in chain(file, [""]):
                line = line.strip()
                if not line and sentence:
                    if len(sentence[0]) > 10:
                        dataset.text_to_instance(sentence, lang)
                    sentence = list()
                elif line.startswith('#'):
                    continue
                else:
                    line = line.split('\t')
                    try:
                        line[0] = int(line[0])
                        sentence.append(line)
                    except ValueError:
                        continue

        return dataset

    def text_to_instance(self, sentence, lang):
        sense_col = 10 if lang == 'en' else 9  # 暂时只考虑英德
        ins = {'words': list(), 'upostag': list(), 'sent': list()}
        pieces, predicate_ids = dict(), list()

        padding_len = len(sentence[0]) - 4
        sentence.insert(0, [0, '[CLS]', '[CLS]', 'X'] + ['_'] * padding_len)
        sentence.append([len(sentence), '[SEP]', '[SEP]', 'X'] + ['_'] * padding_len)

        for i, row in enumerate(sentence):
            ins['upostag'].append(row[3])
            if row[sense_col] != '_':
                predicate_ids.append(i)

            if lang == 'zh':
                # word = row[2]
                word = translate.ToSimplifiedChinese(row[1])
                if word != row[2]:
                    trans = translate.ToSimplifiedChinese(row[2])
                    # print(f"{row[1]}[{word}]: {row[2]}[{trans}]")
                    word = trans
                    assert True
            else:
                word = row[1]
            ins['sent'].append(word)
            if self.tokenizer is not None:
                piece = self.tokenizer.tokenize(word)
                if len(piece) > 0:
                    ins['words'].append(piece[0])
                    if len(piece) > 1:
                        pieces[i] = [self.tokenizer.vocab[p] for p in piece]
            else:
                ins['words'].append(word.lower())

        ins['word_pieces'] = pieces
        if len(self.pretrained_fields) > 0:
            ins["words" + PRETRAIN_POSTFIX] = copy.deepcopy(ins['words'])

        # for row in sentence:
        #     print(len(row), '\t', '\t'.join(row[sense_col:]))
        # print('\n')

        def label_map(label):
            LABEL[label] += 1
            # if label in ('V', 'C-V', 'R-V'):
            #     return '_'  # 待确定
            if 'ARG' in label:
                return label.replace('ARG', 'A')
            return label

        for col, p in enumerate(predicate_ids, start=sense_col + 1):
            one = copy.deepcopy(ins)
            one['indicator'] = p
            one['labels'] = [label_map(line[col]) for line in sentence]
            # if 'V' not in one['labels'][p]:
            #     one['labels'][p] = 'V'  # 10个这种的
            self.data.append(one)

    def collate_fn(self, batch):
        ids_sorted = sorted(
            range(len(batch)), key=lambda i: len(batch[i]['words']), reverse=True)

        max_len = len(batch[ids_sorted[0]]['words'])
        result = defaultdict(lambda: torch.zeros(len(batch), max_len, dtype=torch.long))
        result['seq_lens'] = list()
        result['sentences'] = list()
        result['word_pieces'] = dict()

        for i, origin in zip(range(len(batch)), ids_sorted):
            seq_len = len(batch[origin]['words'])
            result['seq_lens'].append(seq_len)
            result['sentences'].append(batch[origin]['sent'])
            result['mask'][i, :seq_len] = 1
            result['indicator'][i, batch[origin]['indicator']] = 1
            for key in ('words', 'upostag', 'labels'):
                result[key][i, :seq_len] = torch.LongTensor(batch[origin][key])
            for key in self.pretrained_fields:
                result[key][i, :seq_len] = torch.LongTensor(batch[origin][key])
            for w, piece in batch[origin]['word_pieces'].items():
                result['word_pieces'][(i, w)] = torch.LongTensor(piece)
            # if (result['words'][i] == 100).long().sum() > seq_len // 3:
            #     print(batch[0]['metadata']['lang'], ':', batch[origin]['form'])

        return result


class PMBDataset(DataSet):
    """ super-sense dataset """
    index_fields = ("words", "labels", "upostag")

    def __init__(self,
                 data: List,
                 tokenizer: Any = None,
                 pretrained_fields: Set[str] = ()):
        super().__init__(data, tokenizer, pretrained_fields)

    @staticmethod
    def read_all(data_dir: str, lang: str = 'en', fold=8) -> List:
        if os.path.isfile(f"{data_dir}/{fold}-fold.pkl"):
            with codecs.open(f"{data_dir}/{fold}-fold.pkl", mode='rb') as file:
                return pickle.load(file)

        path_list, data = glob.glob(f"{data_dir}/data/{lang}/gold/*/d*/*.xml"), list()
        # pos = set()

        def token_to_dict(token):
            tags = token.find('tags').findall('tag')
            tags = {t.attrib['type']: t.text for t in tags}
            # pos.add(tags.pop('sem', '<unk>'))
            return tags

        for p in path_list:
            root = et.parse(p).getroot()
            tokens = root.find('xdrs').find('taggedtokens').findall('tagtoken')
            tokens = [token_to_dict(t) for t in tokens]
            data.append([t for t in tokens if t['tok'] != 'ø'])

        folds = [list() for _ in range(fold)]
        random.shuffle(data)
        for i, ins in enumerate(data):
            folds[i % fold].append(ins)

        with codecs.open(f"{data_dir}/{fold}-fold.pkl", mode='wb') as file:
            return pickle.dump(folds, file)

        return folds

    @classmethod
    def build(cls,
              data: List,
              tokenizer: Any = None,
              pretrained_fields: Set[str] = (),
              **kwargs):
        dataset = cls(list(), tokenizer, pretrained_fields)
        for ins in data:
            dataset.text_to_instance(ins)
        return dataset

    def text_to_instance(self, sentence):
        ins = {'words': list(), 'upostag': list(), 'sent': list(), 'labels': list()}
        pieces = dict()
        # print('\n')
        sentence.insert(0, {'tok': '[CLS]', 'pos': '<pad>', 'sem': '_'})
        sentence.append({'tok': '[SEP]', 'pos': '<pad>', 'sem': '_'})

        for i, row in enumerate(sentence):
            # print(row)
            ins['sent'].append(row['tok'])
            ins['upostag'].append(row['pos'])
            if self.tokenizer is not None:
                piece = self.tokenizer.tokenize(row['tok'])
                if len(piece) > 0:
                    ins['words'].append(piece[0])
                    if len(piece) > 1:
                        pieces[i] = [self.tokenizer.vocab[p] for p in piece]
            else:
                ins['words'].append(row['tok'].lower())
            ins['labels'].append(row['sem'])

        ins['word_pieces'] = pieces
        if len(self.pretrained_fields) > 0:
            ins["words" + PRETRAIN_POSTFIX] = copy.deepcopy(ins['words'])

        self.data.append(ins)

    def collate_fn(self, batch):
        ids_sorted = sorted(
            range(len(batch)), key=lambda i: len(batch[i]['words']), reverse=True)

        max_len = len(batch[ids_sorted[0]]['words'])
        result = defaultdict(lambda: torch.zeros(len(batch), max_len, dtype=torch.long))
        result['seq_lens'] = list()
        result['sentences'] = list()
        result['word_pieces'] = dict()

        for i, origin in zip(range(len(batch)), ids_sorted):
            seq_len = len(batch[origin]['words'])
            result['seq_lens'].append(seq_len)
            result['sentences'].append(batch[origin]['sent'])
            result['mask'][i, :seq_len] = 1
            for key in ('words', 'upostag', 'labels'):
                result[key][i, :seq_len] = torch.LongTensor(batch[origin][key])
            for key in self.pretrained_fields:
                result[key][i, :seq_len] = torch.LongTensor(batch[origin][key])
            for w, piece in batch[origin]['word_pieces'].items():
                result['word_pieces'][(i, w)] = torch.LongTensor(piece)

        return result


_DATASET = {
    'streusle': STREUSLEDataset,
    'srl': SRLDataset,
    'dep': ConlluDataset,
    'pmb': PMBDataset
}


def build_dataset(name: str, data_dir: str, read_test: bool = False, cache='',
                  **kwargs) -> Dict[str, Any]:
    dataset = dict()
    dataset[KEY_TRAIN] = _DATASET[name].build(data_dir, KEY_TRAIN, **kwargs)
    dataset[KEY_DEV] = _DATASET[name].build(data_dir, KEY_DEV, **kwargs)
    if read_test:
        dataset[KEY_TEST] = _DATASET[name].build(
            data_dir, KEY_TEST, **kwargs)

    return dataset


def index_dataset(dataset, vocab):
    if isinstance(dataset, DataSet):
        dataset.index_dataset(vocab)
    elif isinstance(dataset, dict):
        for key in dataset:
            index_dataset(dataset[key], vocab)
    elif isinstance(dataset, list):
        for i in range(len(dataset)):
            index_dataset(dataset[i], vocab)
