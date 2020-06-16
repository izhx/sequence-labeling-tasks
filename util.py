"""
工具
"""

import os
import codecs
import pickle
from collections import defaultdict

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


def select_vec(dataset, vec_path, new_path):
    counter = defaultdict(int)
    for data in dataset.values():
        for ins in data.data:
            for w in ins['words']:
                counter[w] += 1

    new_vec = []
    with codecs.open(vec_path, mode='r', encoding='UTF-8') as file:
        for line in file.readlines():
            if line.split()[0] in counter:
                new_vec.append(line)

    with codecs.open(new_path, mode='w', encoding='UTF-8') as file:
        file.write(f"{len(new_vec)} 300\n")
        file.writelines(new_vec)

    return
