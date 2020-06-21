# coding=UTF-8

import os
import random
import argparse

import numpy as np
import torch
from torch.optim import Adam
from transformers import BertTokenizer

import nmnlp
from nmnlp.common.util import output
from nmnlp.common.config import Config
from nmnlp.core import Trainer, Vocabulary

from datasets import build_dataset
from models import build_model
from util import read_data, save_data, is_data, select_vec

nmnlp.core.trainer.EARLY_STOP_THRESHOLD = 10

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--yaml', '-y',
                         type=str,
                         default='dep-en',
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
_ARG_PARSER.add_argument('--save', '-s',
                         type=str,
                         default='depsawr',
                         help='存储文件夹名')
_ARGS = _ARG_PARSER.parse_args()


def set_seed(seed: int = 123):
    output(f"Process id: {os.getpid()}, cuda: {_ARGS.cuda}, set seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_once(cfg: Config, vocab, dataset, device, sampler, upostag):
    model = build_model(vocab=vocab, upostag=upostag, **cfg['model'])
    para_num = sum([np.prod(list(p.size())) for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)

    # param_groups = param_groups_with_different_lr(model, 1e-3, bert=1e-4)

    optimizer = Adam(model.parameters(), **cfg['optim'])
    scheduler = None

    if cfg['trainer']['tensorboard'] and _ARGS.debug:
        cfg['trainer']['tensorboard'] = False
    cfg['trainer']['log_batch'] = _ARGS.debug
    trainer = Trainer(cfg, dataset, vocab, model, optimizer, sampler, scheduler,
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
    trainer.test(dataset['test'], 64)

    model.depsawr.save(f'/data/private/zms/model_weight/depsawr-{_ARGS.save}/')


def main():
    """ a   """
    root = "/data/private/zms/sequence-labeling-tasks"
    cfg = Config.from_file(f"{root}/dev/config/{_ARGS.yaml}.yml")
    device = torch.device(f"cuda:{_ARGS.cuda}")  # set_visible_devices(_ARGS.cuda)
    data_kwargs, vocab_kwargs = dict(cfg['data']), dict(cfg['vocab'])
    use_bert = 'bert' in cfg['model']['word_embedding']['name_or_path']

    pos = [
        '<pad>', '<unk>', 'X', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
        'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
        "CONJ"  # de中非upos标准 CONJ
        ]

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

        vocab._token_to_index['upostag'] = {k: i for i, k in enumerate(pos)}
        vocab._index_to_token['upostag'] = {i: k for i, k in enumerate(pos)}

        if use_bert:
            vocab._token_to_index['words'] = tokenizer.vocab
            vocab._index_to_token['words'] = tokenizer.ids_to_tokens

        save_data(data_kwargs['cache'], dataset, vocab)
    else:
        dataset, vocab = read_data(data_kwargs['cache'])

    # select_vec(dataset, "/data/private/zms/DEPSAWR/embeddings/cc.de.300.vec",
    #            f"{root}/dev/vec/cc_de_300_UD.vec")

    run_once(cfg, vocab, dataset, device, None, pos)


if __name__ == '__main__':
    set_seed()
    main()

"""

"""
