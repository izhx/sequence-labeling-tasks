from typing import Tuple, Callable

import codecs
import pickle

import torch

from .config import Configurable
from data.Vocab import Vocab
from .syn import ParserModel


def parser_and_vocab_from_pretrained(dir_path, device) -> Tuple[ParserModel, Vocab]:
    config = Configurable(f"{dir_path}/parser.cfg")
    with codecs.open(f"{dir_path}/model/vocab", mode='rb') as file:

        vocab = pickle.load(file)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    model = ParserModel(vocab, config, vec)
    model.to(device)
    model.load(f"{dir_path}/model/model", device)
    return model, vocab


def collate_fn_wrapper(vocab: Vocab, collate_fn: Callable):
    def new_fn(batch):
        result = collate_fn(batch)
        result['dw'] = torch.zeros_like(result['words'])
        result['ew'] = torch.zeros_like(result['words'])

        for i, sent in enumerate(result['sentences']):
            words = [w.lower() for w in sent]
            result['dw'][i, :len(sent)] = torch.tensor(vocab.word2id(words))
            result['ew'][i, :len(sent)] = torch.tensor(vocab.extword2id(words))

        return result
    return new_fn
