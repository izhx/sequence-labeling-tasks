"""
工具
"""

import os
import codecs
import pickle

from nmnlp.common.util import output
from datasets import index_dataset

PATH = "./dev/cache/"


def is_data(name):
    return os.path.isfile(f"{PATH}data-{name}.bin")


def save_data(name, dataset, vocab, index=False):
    if index:
        index_dataset(dataset, vocab)
    with codecs.open(f"{PATH}data-{name}.bin", 'wb') as f:
        pickle.dump((dataset, vocab), f)
    output(f"===> saved at <{PATH}data-{name}.bin>")


def read_data(name):
    with codecs.open(f"{PATH}data-{name}.bin", 'rb') as f:
        output(f"===> loading from <{PATH}data-{name}.bin>")
        return pickle.load(f)
