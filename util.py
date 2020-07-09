"""
工具
"""

import os
import codecs
import pickle
import sqlite3
from collections import defaultdict

from nmnlp.common.util import output
from nmnlp.core.dataset import DataSet
from datasets import index_dataset

PATH = "./dev/cache/"

DB_PATH = './dev/srlen.db'


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
        if isinstance(data, DataSet):
            data = [data]
        elif isinstance(data, dict):
            data = data.values()
        for one in data:
            for ins in one.data:
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

    output(f"save at <{new_path}>")


def common_query(query):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(query)
        return list(cursor)


def select_number(query, multi=False):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(query)
        if multi:
            return next(cursor)
        return next(cursor)[0]


def cond_num(*args, name='baseline'):
    query, cond = "SELECT SUM(1) FROM ", ""
    if len(args) != 0:
        cond += ' WHERE ' + ' AND '.join(args)
    return select_number(query + name + cond)


def f1(name, *args):
    if len(args) != 0:
        print(' WHERE ' + ' AND '.join(args))
    args = list(args)
    args.extend(["label != '_'", "indicator = 0"])
    all_num = cond_num(*args, name=name)
    args.append("prediction != '_'")
    recall = cond_num(*args, name=name)
    args.append("label = prediction")
    correct = cond_num(*args, name=name)

    p, r = correct / all_num, recall / all_num
    _f1 = 2 * p * r / (p + r)
    print(f"[{name}]: recall {r:.4f}, precision {p:.4f}, f1 {_f1:.4f}")


def main():
    for name in ('baseline', 'depsawr'):
        print('\n')
        f1(name)
        f1(name, 'length < 11')
        f1(name, 'length > 10', 'length < 16')
        f1(name, 'length > 15', 'length < 21')
        f1(name, 'length > 20', 'length < 26')
        f1(name, 'length > 25', 'length < 31')
        f1(name, 'length > 30')


if __name__ == "__main__":
    main()
