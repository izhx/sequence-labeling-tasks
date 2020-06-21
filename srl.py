# coding=UTF-8

import os
import time
import random
import argparse

import numpy as np
import torch
from torch.optim import Adam
from transformers import BertTokenizer

import nmnlp
from nmnlp.common.util import output, set_visible_devices
from nmnlp.common.config import Config
from nmnlp.core import Trainer, Vocabulary
from nmnlp.core.optim import build_optimizer

from notag import parser_and_vocab_from_pretrained, collate_fn_wrapper
from datasets import build_dataset
from models import build_model
from util import read_data, save_data, is_data, select_vec

nmnlp.core.trainer.EARLY_STOP_THRESHOLD = 10

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--yaml', '-y',
                         type=str,
                         default='srlen-depsawr',
                         help='configuration file path.')
_ARG_PARSER.add_argument('--cuda', '-c',
                         type=str,
                         default='3',
                         help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--debug', '-d', type=bool, default=False)
_ARG_PARSER.add_argument("--test", '-t',
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


def run_once(cfg: Config, vocab, dataset, device, parser):
    model = build_model(vocab=vocab, depsawr=parser, **cfg['model'])
    para_num = sum([np.prod(list(p.size())) for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)

    optimizer = build_optimizer(model, **cfg['optim'])

    # epoch_steps = len(dataset['train']) // cfg['trainer']['batch_size'] + 1

    scheduler = None

    if cfg['trainer']['tensorboard'] and _ARGS.debug:
        cfg['trainer']['tensorboard'] = False
    # cfg['trainer']['log_batch'] = _ARGS.debug
    trainer = Trainer(cfg, dataset, vocab, model, optimizer, None, scheduler,
                      device, **cfg['trainer'])

    if 'pre_train_path' in cfg['trainer'] and os.path.isfile(
            cfg['trainer']['pre_train_path']
    ):
        trainer.load()
    else:
        trainer.epoch_start = 0

    if not _ARGS.test:
        trainer.train()
        trainer.load()
    return trainer.test(dataset['test'], 64)


def main():
    """ a   """
    root = "/data/private/zms/sequence-labeling-tasks"
    cfg = Config.from_file(f"{root}/dev/config/{_ARGS.yaml}.yml")

    print(cfg.cfg)

    device = torch.device(f"cuda:{_ARGS.cuda}")  # set_visible_devices(_ARGS.cuda)
    data_kwargs, vocab_kwargs = dict(cfg['data']), dict(cfg['vocab'])
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

    if not is_data(data_kwargs['cache']):
        dataset = build_dataset(**data_kwargs)

        if use_bert:
            vocab_kwargs['create_fields'] = {
                k
                for k in dataset['train'].index_fields if k != 'words'
            }
        else:
            vocab_kwargs['create_fields'] = dataset['train'].index_fields
        vocab = Vocabulary.from_instances(dataset, **vocab_kwargs)

        if 'upostag' in vocab_kwargs['create_fields']:
            pos = [
                'X', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
                "CONJ"  # de中非upos标准 CONJ
            ]
            vocab._token_to_index['upostag'] = {k: i for i, k in enumerate(pos)}
            vocab._index_to_token['upostag'] = {i: k for i, k in enumerate(pos)}

        if use_bert:
            vocab._token_to_index['words'] = tokenizer.vocab
            vocab._index_to_token['words'] = tokenizer.ids_to_tokens

        dataset, vocab = post_process(data_kwargs['name'], dataset, vocab)

        save_data(data_kwargs['cache'], dataset, vocab)
    else:
        dataset, vocab = read_data(data_kwargs['cache'])

    # select_vec(dataset, "/data/private/zms/DEPSAWR/embeddings/cc.de.300.vec",
    #            f"{root}/dev/vec/cc_de_300_UP.vec")

    if 'depsawr' in cfg.cfg:
        parser, parser_vocab = parser_and_vocab_from_pretrained(cfg['depsawr'], device)
        for k in dataset:
            dataset[k].collate_fn = collate_fn_wrapper(parser_vocab, dataset[k].collate_fn)
    else:
        parser = None

    run_once(cfg, vocab, dataset, device, parser)
    loop(device)


def post_process(name, dataset, vocab):
    if name == 'srl':
        labels = [
            'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM-COM', 'AM-LOC', 'AM-DIR', 'AM-GOL',
            'AM-MNR', 'AM-TMP', 'AM-EXT', 'AM-REC', 'AM-PRD', 'AM-PRP', 'AM-CAU',
            'AM-DIS', 'AM-MOD', 'AM-NEG', 'AM-DSP', 'AM-ADV', 'AM-ADJ', 'AM-LVB',
            'AM-CXN', 'AM-PRR', 'A1-DSP', 'V']  # AM-PRR A1-DSP 新的
        labels = labels + ['R-' + i for i in labels] + ['C-' + i for i in labels]
        labels = ['<pad>', '<unk>'] + labels + ['_']
        vocab._token_to_index['labels'] = {k: i for i, k in enumerate(labels)}
        vocab._index_to_token['labels'] = {i: k for i, k in enumerate(labels)}

    return dataset, vocab


def loop(device):
    while True:
        time.sleep(0.01)
        a, b = torch.rand(233, 233, 233).to(device), torch.rand(233, 233, 233).to(device)
        c = a * b
        a = c


if __name__ == '__main__':
    set_seed()
    main()

"""

"""
