"""
工具
"""

import codecs
import pickle

from nmnlp.common.util import output
from dataset import index_dataset

PATH = "./data/"


def save_data(name, dataset, vocab):
    index_dataset(dataset, vocab)
    with codecs.open(f"{PATH}data-{name}.bin", 'wb') as f:
        pickle.dump((dataset, vocab), f)
    output(f"===> saved at <{PATH}data-{name}.bin>")


def read_data(name):
    with codecs.open(f"{PATH}data-{name}.bin", 'rb') as f:
        output(f"===> loading from <{PATH}data-{name}.bin>")
        return pickle.load(f)
