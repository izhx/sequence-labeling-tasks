# coding=UTF-8

import os
import csv
import copy
import time
import codecs
import random
import argparse
from typing import List, cast
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from torch.optim import Adam
from transformers import BertTokenizer

import nmnlp
from nmnlp.common.util import output, set_visible_devices
from nmnlp.common.config import Config
from nmnlp.core import Trainer, Vocabulary
from nmnlp.core.trainer import to_device, format_metric, clip_grad_func
from nmnlp.core.optim import build_optimizer

from notag import parser_and_vocab_from_pretrained, collate_fn_wrapper
from datasets import build_dataset, PMBDataset
from models import build_model
from util import read_data, save_data, is_data, select_vec

nmnlp.core.trainer.EARLY_STOP_THRESHOLD = 10

_ARG_PARSER = argparse.ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--yaml', '-y',
                         type=str,
                         default='srlzh',
                         help='configuration file path.')
_ARG_PARSER.add_argument('--cuda', '-c',
                         type=str,
                         default='0',
                         help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--debug', '-d', type=bool, default=False)
_ARG_PARSER.add_argument("--test", '-t',
                         default=False,
                         action="store_true",
                         help="测试模式，可保存测试结果，或ensemble")
_ARGS = _ARG_PARSER.parse_args()


def set_seed(seed: int = 123):
    output(f"Process id: {os.getpid()}, cuda: {_ARGS.cuda}, set seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_func(self, loader, epoch, step) -> torch.Tensor:
    """ 为了计时而覆盖的训练函数，就用了一次。以后考虑用callback形式。
    """
    losses = torch.zeros(len(loader), device=self.device)
    dep_time = 0
    for i, batch in enumerate(loader):
        model_output = self.model(**to_device(batch, self.device))
        loss = model_output['loss']
        dep_time += model_output['dep_time']
        losses[i] = loss.item()
        if self.update_every == 1:
            loss.backward()
        else:
            (loss / self.update_every).backward()  # gradient accumulation
        if step:
            if self.clip_grad:
                clip_grad_func(self.model.parameters(), **self.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

        if i % self.log_interval == 0 and self.writer:
            n_example = (epoch * len(loader) + i) * loader.batch_size
            self.writer.add_scalar('Train/loss', loss.item(), n_example)
    print(f"===> dep time: {dep_time}")
    return losses


def run_once(cfg: Config, vocab, dataset, device, parser):
    """ 一次训练流程。"""
    model = build_model(vocab=vocab, depsawr=parser, **cfg['model'])
    para_num = sum([np.prod(list(p.size())) for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)

    optimizer = build_optimizer(model, **cfg['optim'])
    scheduler = None

    if cfg['trainer']['tensorboard'] and _ARGS.debug:
        cfg['trainer']['tensorboard'] = False
    # cfg['trainer']['log_batch'] = _ARGS.debug
    trainer = Trainer(cfg, dataset, vocab, model, optimizer, None, scheduler,
                      device, **cfg['trainer'])
    # trainer.train_func = train_func  # 为了计时

    # 训练过程
    trainer.train()
    trainer.load()  # 加载配置文件中给定path的checkpoint，存档模式为best时有意义
    return trainer.test(dataset['test'], 64)


def tensor_avg(*tensors):
    return sum(tensors) / len(tensors)


def tensor_max(*tensors):
    tensors = list(tensors)
    for i in range(len(tensors)):
        tensors[i] = tensors[i].unsqueeze(0)
    all_max = torch.cat(tensors, dim=0).max(dim=0)[0]
    return all_max


def test_result(cfg: Config, vocab, dataset, device, parser, ensemble_path=None):
    model = build_model(vocab=vocab, depsawr=parser, **cfg['model'])
    para_num = sum([np.prod(list(p.size())) for p in model.parameters()])
    output(f'param num: {para_num}, {para_num / 1000000:4f}M')
    model.to(device=device)
    optimizer = build_optimizer(model, **cfg['optim'])
    trainer = Trainer(cfg, dataset, vocab, model, optimizer, None, None,
                      device, **cfg['trainer'])
    trainer.load()

    tensor_op = tensor_avg
    if ensemble_path and os.path.isfile(ensemble_path):
        # 如果给了ensemble_path，将其加载
        another = build_model(vocab=vocab, depsawr=parser, **cfg['model'])
        another.to(device=device)
        checkpoint = torch.load(ensemble_path, map_location=device)
        another.load_state_dict(checkpoint['model'])
        another.test_mode(device)
        print(f"===> model loaded from <{ensemble_path}>")

        # 发射概率矩阵也按策略变换
        tran = tensor_op(model.crf.transitions.data, another.crf.transitions.data)
        model.crf.transitions.data = tran
    else:
        another = None

    table = [['sentence_id', 'length', 'word_id', 'word', 'pos', 'indicator', 'label', 'prediction']]

    def process_one(self, one_set, name, device, batch_size, epoch=None):
        """ epoch is None means test stage.
        """
        loader = self.get_loader(one_set, batch_size)
        len_loader = len(loader)
        losses = torch.zeros(len_loader, device=device)

        for i, batch in enumerate(loader):
            batch = to_device(batch, device)
            model_output = self.model(**batch)
            losses[i] = model_output['loss'].item()
            if another:
                # ensemble预测
                scores = another(**batch)['scores']
                scores = tensor_op(scores, model_output['scores'])
                best_paths = self.model.crf.viterbi_tags(scores, batch['mask'], 1)
                model_output['predicted_tags'] = cast(List[List[int]], [x[0][0] for x in best_paths])

            # 记录测试结果
            for j, predicted in enumerate(model_output['predicted_tags']):
                sid = i * batch_size + j
                length = batch['seq_lens'][j]
                for n, word in enumerate(batch['sentences'][j]):
                    pos = vocab.index_to_token(batch['upostag'][j, n].item(), 'upostag')
                    label = vocab.index_to_token(batch['labels'][j, n].item(), 'labels')
                    indicator = batch['indicator'][j, n].item()
                    prediction = vocab.index_to_token(predicted[n], 'labels')
                    table.append([sid, length, n, word, pos, indicator, label, prediction])

        metric_counter = copy.deepcopy(self.model.metric.counter)
        metric = self.model.get_metrics(reset=True)
        if epoch is not None and self.writer is not None:
            metric['loss'] = losses.mean()
            self.add_scalars('Very_Detail', metric, epoch, name)
            self.writer.flush()
        elif epoch is None:
            output(f"Test {name} compete, {format_metric(metric)}")
        return metric_counter, metric, losses

    trainer.process_one = process_one  # 覆盖trainer处理函数
    trainer.test(dataset['test'], 64)

    with codecs.open(f"./dev/result/{trainer.prefix}.csv", mode='w', encoding='UTF-8') as file:
        writer = csv.writer(file)
        writer.writerows(table)
    output(f"saved <./dev/result/{trainer.prefix}.csv>")


def main():
    """ a   """
    root = "/data/private/zms/sequence-labeling-tasks"
    cfg = Config.from_file(f"{root}/dev/config/{_ARGS.yaml}.yml")

    cfg['trainer']['epoch_start'] = 0

    print(cfg.cfg)  # 配置文件

    device = torch.device(f"cuda:{_ARGS.cuda}")  # set_visible_devices(_ARGS.cuda)
    data_kwargs, vocab_kwargs = dict(cfg['data']), dict(cfg['vocab'])
    use_bert = 'bert' in cfg['model']['word_embedding']['name_or_path']

    # 如果用了BERT，要加载tokenizer
    if use_bert:
        tokenizer = BertTokenizer.from_pretrained(
            cfg['model']['word_embedding']['name_or_path'],
            do_lower_case=False)
        print("I'm batman!  ",
              tokenizer.tokenize("I'm batman!"))  # [CLS] [SEP]
        data_kwargs['tokenizer'] = tokenizer
        vocab_kwargs['oov_token'] = tokenizer.unk_token,
        vocab_kwargs['padding_token'] = tokenizer.pad_token
    else:
        tokenizer = None

    if not is_data(data_kwargs['cache']):
        # 如果没有cache，重新读取数据并创建cache
        if data_kwargs['name'] == 'pmb':
            # pmb的数据是单独处理，分8折
            folds = PMBDataset.read_all(data_kwargs['data_dir'])
            dataset = {
                'train': PMBDataset.build(chain(*folds[:-2]), **data_kwargs),
                'dev': PMBDataset.build(folds[-2], **data_kwargs),
                'test': PMBDataset.build(folds[-1], **data_kwargs)
            }
        else:
            dataset = build_dataset(**data_kwargs)

        if use_bert:
            # 用了BERT则不用对words建立词表
            vocab_kwargs['create_fields'] = {
                k
                for k in dataset['train'].index_fields if k != 'words'
            }
        else:
            vocab_kwargs['create_fields'] = dataset['train'].index_fields

        # 建立词表
        vocab = Vocabulary.from_instances(dataset, **vocab_kwargs)

        if 'upostag' in vocab_kwargs['create_fields']:
            # 将upostag词表人为的填充完整，数据里可能不全
            pos = [
                'X', 'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
                "CONJ"  # de中非upos标准 CONJ
            ]
            vocab._token_to_index['upostag'] = {k: i for i, k in enumerate(pos)}
            vocab._index_to_token['upostag'] = {i: k for i, k in enumerate(pos)}

        if use_bert:
            # 若用BERT，则把words词表替换为BERT的
            vocab._token_to_index['words'] = tokenizer.vocab
            vocab._index_to_token['words'] = tokenizer.ids_to_tokens

        # 其他后处理，比如标签词表的整理
        dataset, vocab = post_process(data_kwargs['name'], dataset, vocab)

        save_data(data_kwargs['cache'], dataset, vocab)
    else:
        dataset, vocab = read_data(data_kwargs['cache'])

    # 用词表筛选出预训练词向量中出现过的，这里生成新vec后，要删掉上面数据cache，注释掉下面
    # 每个任务跑一次select_vec就行。这里还是不太合理，以后调整
    # select_vec(dataset, "/data/private/zms/DEPSAWR/embeddings/glove.6B.300d.txt",
    #            f"{root}/dev/vec/glove_6B_300d_SEM.vec")

    if 'depsawr' in cfg.cfg:
        # 读取预训练depsawr，把数据的collate function重新包装，
        parser, parser_vocab = parser_and_vocab_from_pretrained(cfg['depsawr'], device)
        for k in dataset:
            dataset[k].collate_fn = collate_fn_wrapper(parser_vocab, dataset[k].collate_fn)
        if not cfg['trainer']['prefix'].endswith('dep'):
            cfg['trainer']['prefix'] += '-dep'
    else:
        parser = None

    if _ARGS.test:
        # 仅测试模式，读取checkpoint，测试并保存结果
        # 需要ensemble的时候取消下方几行注释，给定checkpoint path
        ensemble_path = None
        # if _ARGS.yaml == 'srlen0':
        #     ensemble_path = 'dev/model/ens-1_16.bac'
        # elif _ARGS.yaml == 'srlen-dep0':
        #     ensemble_path = 'dev/model/ens-dep1_25.bac'
        test_result(cfg, vocab, dataset, device, parser, ensemble_path)
    else:
        # 训练
        run_once(cfg, vocab, dataset, device, parser)
        # 训练结束后进入挂机死循环
        loop(device)


def post_process(name, dataset, vocab):
    if name == 'srl':
        labels = [
            'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM-COM', 'AM-LOC', 'AM-DIR', 'AM-GOL',
            'AM-MNR', 'AM-TMP', 'AM-EXT', 'AM-REC', 'AM-PRD', 'AM-PRP', 'AM-CAU',
            'AM-DIS', 'AM-MOD', 'AM-NEG', 'AM-DSP', 'AM-ADV', 'AM-ADJ', 'AM-LVB',
            'AM-CXN', 'AM-PRR', 'A1-DSP', 'V']  # AM-PRR A1-DSP 新的
        labels = labels + ['R-' + i for i in labels] + ['C-' + i for i in labels]
        labels = ['<pad>', '<unk>'] + labels + ['O_`$', '_']
    elif name == 'streusle':
        labels = [
            "n.relation", "n.cognition", "n.group", "n.feeling", "n.animal",
            "n.person", "n.time", "n.plant", "n.event", "n.artifact", "n.state",
            "n.quantity", "n.phenomenon", "n.process", "n.naturalobject",
            "n.act", "n.communication", "n.other", "n.attribute", "n.shape",
            "n.food", "n.motive", "n.body", "n.location", "n.substance",
            "n.possession", "p.interval", "p.instrument", "p.locus",
            "p.originator", "p.rateunit", "p.comparisonref", "p.theme",
            "p.socialrel", "p.path", "p.time", "p.whole", "p.stuff",
            "p.ensemble", "p.topic", "p.starttime", "p.causer", "p.possessor",
            "p.experiencer", "p.orgmember", "p.beneficiary", "p.direction",
            "p.means", "p.partportion", "p.recipient", "p.approximator",
            "p.cost", "p.stimulus", "p.explanation", "p.species", "p.goal",
            "p.agent", "p.possession", "p.duration", "p.source", "p.extent",
            "p.circumstance", "p.endtime", "p.identity", "p.quantityitem",
            "p.org", "p.characteristic", "p.manner", "p.frequency",
            "p.ancillary", "p.purpose", "p.gestalt", "v.creation", "v.motion",
            "v.change", "v.body", "v.contact", "v.competition", "v.cognition",
            "v.social", "v.communication", "v.stative", "v.perception",
            "v.emotion", "v.possession", "v.consumption", '??'
        ]
        labels = ['B_' + i for i in labels] + ['I_' + i for i in labels] + ['O_' + i for i in labels]
        labels = ['<pad>', '<unk>'] + labels + ['O_`$', '_']
    elif name == 'pmb':
        pos = [
            'TO', 'RQU', 'FW', 'CD', 'JJS', 'POS', 'WRB', 'RP', 'NN', 'NNS',
            ',', 'CC', 'SO', 'VBZ', 'LRB', 'DT', 'NNPS', 'IN', 'RRB', 'NNP',
            'VBG', 'WP$', 'PRP', '.', 'EX', 'JJ', 'UH', 'VBD', 'WP', ':',
            'RBS', 'MD', 'RB', 'VBN', 'VBP', 'PDT', '$', 'PRP$', ';', 'JJR',
            'VB', 'WDT', 'LQU']
        pos = ['<pad>', '<unk>'] + pos
        vocab._token_to_index['upostag'] = {k: i for i, k in enumerate(pos)}
        vocab._index_to_token['upostag'] = {i: k for i, k in enumerate(pos)}
        labels = [
            'PRO', 'DST', 'EXT', 'GRE', 'IST', 'GPE', 'SCO', 'EXS', 'GEO', 'DIS',
            'COL', 'LES', 'MOY', 'CLO', 'NOW', 'ORG', 'NOT', 'INT', 'DOW', 'NIL',
            'NEC', 'CON', 'BUT', 'REF', 'DEG', 'COO', 'EXG', 'ART', 'PRX', 'UOM',
            'TOP', 'QUV', 'ALT', 'YOC', 'EQU', 'SUB', 'SST', 'QUE', 'MOR', 'FUT',
            'ORD', 'PFT', 'LIT', 'PER', 'ENS', 'POS', 'ROL', 'PRG', 'QUC', 'IMP',
            'DEF', 'APX', 'HAS', 'REL', 'GRP', 'XCL', 'HAP', 'EPS', 'PST', 'EFS',
            'GPO', 'NTH', 'ITJ', 'DOM', 'EMP', 'AND', 'BOT', 'CTC']
        labels = ['<pad>', '<unk>'] + labels + ['_']

    vocab._token_to_index['labels'] = {k: i for i, k in enumerate(labels)}
    vocab._index_to_token['labels'] = {i: k for i, k in enumerate(labels)}
    return dataset, vocab


def loop(device):
    output("start looping...")
    while True:
        time.sleep(0.05)
        a, b = torch.rand(233, 233, 233).to(device), torch.rand(233, 233, 233).to(device)
        c = a * b
        a = c


if __name__ == '__main__':
    set_seed()
    main()

"""

"""
