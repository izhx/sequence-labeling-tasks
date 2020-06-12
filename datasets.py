"""
数据集
"""

import copy
import glob
import codecs
from typing import Dict, List, Any, Set
from itertools import chain
from collections import defaultdict

import torch

from nmnlp.common.constant import KEY_TRAIN, KEY_DEV, KEY_TEST
from nmnlp.core.dataset import DataSet


class STREUSLEDataset(DataSet):
    """ super-sense dataset """
    index_fields = ("words",)

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
        path = f"{data_dir}/{kind}/streusle.ud_{kind}.conllulex"
        data = []

        with codecs.open(path, mode='r', encoding='UTF-8') as file:
            sentence = list()
            for line in chain(file, [""]):
                line = line.strip()
                if not line and sentence:
                    data.append(cls.text_to_instance(sentence))
                    sentence = list()
                elif line.startswith('#'):
                    continue
                else:
                    sentence.append(line)

        return cls(data, tokenizer, pretrained_fields)

    def text_to_instance(self, sentence):
        raise NotImplementedError

    def collate_fn(self, batch):
        raise NotImplementedError


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
        sentence.insert(0, [0, '[CLS]', '', 'X'] + ['_'] * padding_len)
        sentence.append([len(sentence), '[SEP]', '', 'X'] + ['_'] * padding_len)

        for i, row in enumerate(sentence):
            ins['sent'].append(row[1])
            ins['upostag'].append(row[3])
            if row[sense_col] != '_':
                predicate_ids.append(i)
            if self.tokenizer is not None:
                piece = self.tokenizer.tokenize(row[1])
                if len(piece) > 0:
                    ins['words'].append(piece[0])
                    if len(piece) > 1:
                        pieces[i] = [self.tokenizer.vocab[p] for p in piece]
            else:
                ins['words'].append(row[1])
        ins['sent'] = ' '.join(ins['sent'])
        ins['word_pieces'] = pieces

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
        if self.tokenizer is not None:
            max_len += 1 # for bert
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
            if self.tokenizer is not None:
                result['words'][i, 0] = 101  # [CLS]
                result['words'][i, seq_len] = 102  # [SEP]
            # if (result['words'][i] == 100).long().sum() > seq_len // 3:
            #     print(batch[0]['metadata']['lang'], ':', batch[origin]['form'])

        return result


_DATASET = {
    'streusle': STREUSLEDataset,
    'srl': SRLDataset,
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
