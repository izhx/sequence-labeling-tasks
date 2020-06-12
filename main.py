"""
多语言句法分析
"""

import os
import argparse
import random
import pickle

import numpy as np
import torch
from torch.optim import Adam
from transformers import BertTokenizer

import nmnlp
from nmnlp.common.util import output, set_visible_devices
from nmnlp.common.config import Config
from nmnlp.core import Trainer, Vocabulary
# from nmnlp.models import CRFTagger

from dataset import STREUSLEDataset, build_dataset
from util import read_data, save_data

nmnlp.core.trainer.EARLY_STOP_THRESHOLD = 20

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('-yaml',
                         type=str,
                         default='./srlen.yml',
                         help='configuration file path.')
_ARG_PARSER.add_argument('-cuda',
                         type=str,
                         default='0',
                         help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('-debug', type=bool, default=False)
_ARG_PARSER.add_argument("-test",
                         default=False,
                         action="store_true",
                         help="test mode")
_ARGS = _ARG_PARSER.parse_args()


def set_seed(seed: int = 123):
    output(f"Process id: {os.getpid()}, cuda: {_ARGS.cuda}, set seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# def run_once(cfg: Config, vocab, dataset, device, sampler):
#     loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
#     model = CRFTagger(loss, vocab, **cfg['model'])
#     para_num = sum([np.prod(list(p.size())) for p in model.parameters()])
#     output(f'param num: {para_num}, size: {para_num * 4 / 1000 / 1000:4f}M')
#     model.to(device=device)

#     # param_groups = param_groups_with_different_lr(model, 1e-3, bert=1e-4)

#     optimizer = Adam(model.parameters(), **cfg['optim'])
#     scheduler = None

#     if cfg['trainer']['tensorboard'] and _ARGS.debug:
#         cfg['trainer']['tensorboard'] = False
#     cfg['trainer']['log_batch'] = _ARGS.debug
#     trainer = Trainer(cfg, dataset, vocab, model, optimizer, sampler, scheduler,
#                       device, **cfg['trainer'])

#     if 'pre_train_path' in cfg['trainer'] and os.path.isfile(
#             cfg['trainer']['pre_train_path']
#     ):
#         trainer.load()
#     else:
#         trainer.epoch_start = 0
#     if _ARGS.lang is not None:
#         trainer.prefix += _ARGS.lang

#     if not _ARGS.test:
#         trainer.train()
#         trainer.load()
#     return trainer.test(dataset['test'], 64)


def main():
    """ a   """

    cfg = Config.from_file(_ARGS.yaml)
    device = set_visible_devices(_ARGS.cuda)
    data_kwargs, vocab_kwargs = {}, dict(cfg['vocab'])
    use_bert = 'bert' in cfg['model']['word_embedding']['name_or_path']

    if use_bert:
        tokenizer = BertTokenizer.from_pretrained(
            cfg['model']['word_embedding']['name_or_path'],
            do_lower_case=False)
        print("I'm batman!  ",
              tokenizer.tokenize("I'm batman!"))  # [CLS] [SEP]
        data_kwargs['tokenizer'] = tokenizer
        vocab_kwargs['oov_token'] = tokenizer.unk_token,
        vocab_kwargs['padding_token'] = tokenizer.pad_token

    # dataset, vocab = read_data('streusle')

    dataset = build_dataset(name='srl',
                            data_dir='./UniversalPropositions/UP_English-EWT/',  # German
                            lang='en',
                            read_test=True,
                            **data_kwargs)

    if use_bert:
        vocab_kwargs['create_fields'] = {
            k
            for k in dataset['train'].index_fields if k != 'words'
        }
    else:
        vocab_kwargs['create_fields'] = dataset['train'].index_fields
    vocab = Vocabulary.from_instances(dataset, **vocab_kwargs)

    labels = [
        'A0', 'A1', 'A2', 'A3', 'A4', 'AM-COM', 'AM-LOC', 'AM-DIR', 'AM-GOL',
        'AM-MNR', 'AM-TMP', 'AM-EXT', 'AM-REC', 'AM-PRD', 'AM-PRP', 'AM-CAU',
        'AM-DIS', 'AM-MOD', 'AM-NEG', 'AM-DSP', 'AM-ADV', 'AM-ADJ', 'AM-LVB',
        'AM-CXN']
    labels = labels + ['R-' + i for i in labels] + ['C-' + i for i in labels]
    labels.append['_']
    vocab._token_to_index['labels'] = {k: i for i, k in enumerate(labels)}
    vocab._index_to_token['labels'] = {i: k for i, k in enumerate(labels)}

    if use_bert:
        vocab._token_to_index['words'] = tokenizer.vocab
        vocab._index_to_token['words'] = tokenizer.ids_to_tokens

    # save_data('bert')

    # run_once(cfg, vocab, dataset, device, None)
    # input("Press ENTER to exit.")
    # print('ENTER')


if __name__ == '__main__':
    set_seed()
    main()
